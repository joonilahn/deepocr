# fmt: off
# define model
_base_ = [
    "../_base_/models/aster.py",
    "../_base_/datasets/kocr_crossentropy.py",
    "../_base_/schedules/schedule.py",
    "../_base_/default_runtime.py",
]

# set work_dir
work_dir = "./kocr_recognizer/work_dir/kocr_aster"
