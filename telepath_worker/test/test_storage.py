from telepath_worker import storage_layer

s3loc = 's3://example-telepath-bucket/test/123.txt'
non_s3loc = '/tmp/test/123.txt'


def _test_s3_storage():
    storage_layer.test_write_df_to_s3_location() # expensive


def test_s3_parse():
    bucket, key = storage_layer.parse_s3_location(s3loc)
    print ( bucket, key )
    assert bucket=='telepath-01'
    assert key=='test/123.txt'

def test_check():
    assert storage_layer.is_s3_location(s3loc)==True
    assert storage_layer.is_s3_location(non_s3loc)==False


if __name__=="__main__":
    test_check()
