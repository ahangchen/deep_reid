import scipy.io
from sklearn import manifold

from utils.file_helper import safe_mkdir
from utils.serialize import pickle_load
import matplotlib as mpl
import matplotlib.style

mpl.style.use('classic')
mpl.use('agg')
import matplotlib.pyplot as plt


def viz_per_camera(feature_path, track_path):
    result = scipy.io.loadmat(feature_path)
    features = result['ft']
    print('feature shape', features.shape)
    cameras_tracks = pickle_load(track_path)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    for ci, camera_track in enumerate(cameras_tracks):
        x = []
        y = []
        info_size = len(camera_track[0])
        print('collect %d infos for c%d' % (info_size, ci + 1))
        for info in camera_track[0][: info_size // 2]:
            y.append(info[0])
            x.append(features[info[2]])
        print('tsne')
        x_tsne = tsne.fit_transform(x)
        print('norm')
        x_norm = (x_tsne - x_tsne.min(0)) / (x_tsne.max(0) - x_tsne.min(0))
        print('plot')
        f = plt.figure(figsize=(10, 10))
        for i in range(x_norm.shape[0]):
            plt.text(x_norm[i, 0], x_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i] % 10),
                     fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig('c%d.pdf' % (ci + 1))


def viz_per_tracklet(dataset, feature_path, track_path):
    safe_mkdir(dataset)
    result = scipy.io.loadmat(feature_path)
    features = result['ft']
    print('feature shape', features.shape)
    cameras_tracks = pickle_load(track_path)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    plt.figure(figsize=(6, 6))
    colors = ['#EE0000', '#00EE00', '#0000EE',
              '#EEEE00', '#EE00EE', '#00EEEE',
              '#EE8800', '#EE0088', '#0088EE', '#8800EE', '#00EE88', '#88EE00']

    for ci, camera_track in enumerate(cameras_tracks):
        if ci > 2:
            break
        x = []
        y = []
        z = []
        info_size = len(camera_track[0])
        print('collect %d infos for c%d' % (info_size, ci + 1))
        frame_idx = 0
        last_track_id = 0
        while frame_idx < info_size and last_track_id < 10:
            info = camera_track[0][frame_idx]
            if info[3] != last_track_id:
                if len(x) > 1:
                    print('tsne on c%d track %d' % (ci + 1, last_track_id))
                    x_tsne = tsne.fit_transform(x)
                    print('norm')
                    x_norm = (x_tsne - x_tsne.min(0)) / (x_tsne.max(0) - x_tsne.min(0))
                    print('plot')
                    color_map = {}
                    color_use_cnt = 0
                    for i in range(x_norm.shape[0]):
                        if not str(y[i]) in color_map:
                            color_use_cnt += 1
                            color_map[str(y[i])] = color_use_cnt
                        plt.text(x_norm[i, 0], x_norm[i, 1], str(y[i]), color=colors[color_map[str(y[i])] % 12],
                                 fontdict={'size': 11})
                        plt.scatter(x_norm[i, 0] + 0.02, x_norm[i, 1] - 0.025, c=colors[color_map[str(y[i])] % 12], linewidths=0., s=100)
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(dataset + '/c%dt%d.pdf' % (ci + 1, last_track_id))
                    plt.clf()
                    print('plot')
                    color_map = {}
                    color_use_cnt = 0
                    for i in range(x_norm.shape[0]):
                        if not str(z[i]) in color_map:
                            color_use_cnt += 1
                            color_map[str(z[i])] = color_use_cnt
                        plt.text(x_norm[i, 0], x_norm[i, 1], str(z[i]), color=colors[color_map[str(z[i])] % 12],
                                 fontdict={'size': 11})
                        plt.scatter(x_norm[i, 0] + 0.02, x_norm[i, 1] - 0.025, c=colors[color_map[str(z[i])] % 12], linewidths=0., s=100)
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(dataset + '/c%dp%d.pdf' % (ci + 1, last_track_id))
                    plt.clf()
                del x[:]
                del y[:]
                del z[:]
            y.append(info[0])
            z.append(info[-1])
            x.append(features[info[2]])
            last_track_id = info[3]
            if last_track_id > 6:
                break
            frame_idx += 1


if __name__ == '__main__':
    cameras_tracks = pickle_load('/home/cwh/coding/TrackViz/pre_process/duke_market_cluster.pck')
    # viz_per_tracklet('duke', 'eval/market_duke-train/train_ft.mat', '/home/cwh/coding/TrackViz/pre_process/spectral.pck')
    viz_per_tracklet('market', 'eval/duke_market-train/train_ft.mat',
                     '/home/cwh/coding/TrackViz/pre_process/market_cluster.pck')
