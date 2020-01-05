import zipfile as zf

def deCrypt(zipfile , outputdir, password):
    """
    使用之前先创造ZipObj类
    解压文件
    """
    zfile = zf.ZipFile(zipfile)
    zfile.extractall( path = outputdir, pwd = password.encode('utf-8')) #encode('utf-8')


def read_zip(file_name,extranct_file , password):
    zip_file = file_name + '.zip'
    print(zip_file)
    with zf.ZipFile(zip_file,'r') as z:
        f = z.open(extranct_file , pwd = password.encode('utf-8'))
        for line in f:
            print(line)

if __name__ == "__main__":
    data_root = '../data/translate/small/'
    deCrypt(data_root + 'clean3.en.zip' , data_root + 'unzip','brucelvjasmineqiu')
    read_zip(data_root + 'clean3.en','clean3.en','brucelvjasmineqiu')