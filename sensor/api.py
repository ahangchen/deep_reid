def singleton(cls, *args, **kw):
    instances = {}

    def _singleton():
        if cls not in instances:
            instances[cls] = cls(*args, **kw)
        return instances[cls]

    return _singleton


@singleton
class ApiConfig(object):
    backend_host = 'http://222.201.145.237:8081'
    file_svr_host = 'http://222.201.145.237:8080/reid'
    urls = {
        'upload_wifi_info': backend_host + '/wifi/record',
        'upload_detect_info': backend_host + '/vision/record',
        'upload_img': backend_host + '/file/img',
        'img_svr': file_svr_host + '/img'
    }
    def __init__(self, x=0):
        self.x = x  
