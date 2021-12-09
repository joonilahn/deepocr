import os.path as osp
import pickle
import shutil
import tempfile
import time

import torch
import mmcv
import torch.distributed as dist
from mmcv.runner import get_dist_info

from ..datasets.dataset_wrappers import ConcatDataset
from ..datasets.pipelines import SimpleTestAug


def single_gpu_test(model, data_loader, return_acc=True, return_filenames=False, max_num=None):
    """A function to test model with dataloader with single gpu.
    
    Args:
        model (MMDataParallel): A PyTorch model wrapped with MMDataParallel.
        data_loader (torch.util.data.DataLoader): A dataloader for a test dataset.
        return_acc (bool, optional):
            return_acc should be true only if a key, 'gt_label', exists in the dataset,
            This is typically enabled for validation dataset.
            For a single inference with no gt_label, set to False.
        max_num (int, optional): This limits the number of test data. Defaults to None.
    
    Returns:
        results (list[dict]): Each result is dict type.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    if max_num is not None:
        prog_bar = mmcv.ProgressBar(max_num)
    else:
        prog_bar = mmcv.ProgressBar(len(dataset))

    single_data = False

    if isinstance(dataset, ConcatDataset):
        transforms = dataset.datasets[0].pipeline.transforms
    else:
        transforms = dataset.pipeline.transforms

    for t in transforms:
        if isinstance(t, SimpleTestAug):
            single_data = True

    for i, data in enumerate(data_loader):
        if return_acc:
            gt_label = data.pop("gt_label")

            if return_filenames:
                filenames = [
                    osp.basename(d["filename"]) for d in data["img_metas"].data[0]
                ]

            if single_data:
                gt_label = gt_label[0]
                batch_size = 1
            else:
                batch_size = gt_label.size(0)
        else:
            gt_label = None

        with torch.no_grad():
            result = model(return_loss=False, **data)

        if return_acc:
            if return_filenames:
                results.append(
                    dict(result=result, gt_label=gt_label, filenames=filename)
                )
            else:
                results.append(dict(result=result, gt_label=gt_label))
        else:
            results.append(dict(result=result))

        for _ in range(batch_size):
            prog_bar.update()

        if max_num is not None:
            accum_data = (i + 1) * batch_size
            if accum_data > max_num:
                break

    return results

def multi_gpu_test(model, data_loader, return_acc=True, return_filenames=False, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting
    ``gpu_collect=True``, it encodes results to gpu tensors and use gpu
    communication for results collection. On cpu mode it saves the results on
    different gpus to ``tmpdir`` and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        if return_acc:
            gt_label = data.pop("gt_label")
            if return_filenames:
                filenames = [
                    osp.basename(d["filename"]) for d in data["img_metas"].data[0]
                ]
        else:
            gt_label = None

        with torch.no_grad():
            result = model(return_loss=False, **data)

        if return_acc:
            if return_filenames:
                results.append(
                    dict(result=result, gt_label=gt_label, filenames=filenames)
                )
            else:
                results.append(dict(result=result, gt_label=gt_label))
        else:
            results.append(dict(result=result))

        if rank == 0:
            batch_size = gt_label.size(0)
            for _ in range(batch_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    """Collect results under cpu mode.
    On cpu mode, this function will save the results on different gpus to
    ``tmpdir`` and collect them by the rank 0 worker.
    Args:
        result_part (list): Result list containing result parts
            to be collected.
        size (int): Size of the results, commonly equal to length of
            the results.
        tmpdir (str | None): temporal directory for collected results to
            store. If set to None, it will create a random temporal directory
            for it.
    Returns:
        list: The collected results.
    """
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_result = mmcv.load(part_file)
            # When data is severely insufficient, an empty part_result
            # on a certain gpu could makes the overall outputs empty.
            if part_result:
                part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    """Collect results under gpu mode.
    On gpu mode, this function will encode results to gpu tensors and use gpu
    communication for results collection.
    Args:
        result_part (list): Result list containing result parts
            to be collected.
        size (int): Size of the results, commonly equal to length of
            the results.
    Returns:
        list: The collected results.
    """
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_result = pickle.loads(recv[:shape[0]].cpu().numpy().tobytes())
            # When data is severely insufficient, an empty part_result
            # on a certain gpu could makes the overall outputs empty.
            if part_result:
                part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results