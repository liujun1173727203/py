import os



from .dataset_dectect import MVTecDataset
def get_training_data(rgb_dir, img_options, class_name):
    assert os.path.exists(rgb_dir)
    return MVTecDataset( dataset_path=rgb_dir, resize=img_options,class_name=class_name)#256  img_options输入图片的大小


def get_validation_data(rgb_dir, img_options,class_name):
    assert os.path.exists(rgb_dir)
    return MVTecDataset(dataset_path=rgb_dir, resize=img_options, class_name=class_name, is_val=True)


def get_test_data(rgb_dir, img_options,class_name):
    assert os.path.exists(rgb_dir)
    return MVTecDataset(dataset_path=rgb_dir, resize=img_options,class_name=class_name, is_train= False)
