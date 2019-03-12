import os
from shutil import copyfile


def img2iddir(src_dir, dst_dir):
    if not os.path.exists(src_dir):
        return
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    for root, dirs, files in os.walk(src_dir, topdown=True):
        for name in files:
            if name.split('.')[-1] not in ['jpg', 'jpeg', 'bmp', 'png'] :
                continue
            ID = name.split('_')
            src_path = src_dir + '/' + name
            dst_path = dst_dir + '/' + ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)


# You only need to change this line to your dataset download path
def prepare_pytorch_ds(download_path='/home/cwh/coding/Market'):
    if not os.path.isdir(download_path):
        print('please change the download_path')

    save_path = download_path + '/pytorch'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    #-----------------------------------------
    #query
    print('on query')
    query_path = download_path + '/probe'
    query_save_path = download_path + '/pytorch/query'
    img2iddir(query_path, query_save_path)
    #-----------------------------------------
    #gallery
    print('on gallery')
    gallery_path = download_path + '/test'
    gallery_save_path = download_path + '/pytorch/gallery'
    img2iddir(gallery_path, gallery_save_path)
    #---------------------------------------
    #train_all
    print('on train all')
    train_path = download_path + '/train'
    train_save_path = download_path + '/pytorch/train_all'
    img2iddir(train_path, train_save_path)

    #---------------------------------------
    #train_val
    print('on train val')
    train_path = download_path + '/train'
    train_save_path = download_path + '/pytorch/train'
    val_save_path = download_path + '/pytorch/val'
    if not os.path.isdir(train_save_path):
        os.mkdir(train_save_path)
        os.mkdir(val_save_path)

    for root, dirs, files in os.walk(train_path, topdown=True):
        for name in files:
            if name.split('.')[-1] not in ['jpg', 'jpeg', 'bmp', 'png'] :
                continue
            ID  = name.split('_')
            src_path = train_path + '/' + name
            dst_path = train_save_path + '/' + ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
                dst_path = val_save_path + '/' + ID[0]  #first image is used as val image
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)


def main(source):
    if source == 'market':
        prepare_pytorch_ds('/home/cwh/coding/dataset/market')
    elif source == 'duke':
        prepare_pytorch_ds('/home/cwh/coding/dataset/duke')
    elif source == 'grid':
        prepare_pytorch_ds('/home/cwh/coding/dataset/grid')
    elif source == 'viper':
        prepare_pytorch_ds('/home/cwh/coding/dataset/viper')
    elif source == 'cuhk':
        prepare_pytorch_ds('/home/cwh/coding/dataset/cuhk01')
    elif source == 'grid-cv':
        for i in range(10):
            prepare_pytorch_ds('/home/cwh/coding/grid_train_probe_gallery/cross%d' % i)

if __name__ == '__main__':
    # sources = ['duke', 'grid', 'viper', 'cuhk', 'grid-cv']
    sources = ['market']
    for source in sources:
        print('processing', source)
        main(source)
