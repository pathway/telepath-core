import uuid

def get_rand_sampleid(length=16):
    return get_rand_id('S',length)

def get_rand_trajid(length=16):
    return get_rand_id('J',length)

def get_rand_agentid(length=16):
    return get_rand_id('A',length)

def get_rand_execid(length=16):
    return get_rand_id('X',length)

def get_rand_id(prefix=None, length=16):
    r = uuid.uuid4().hex[:length]
    if prefix:
        r = prefix+r
    return r

def dbg( section, comment='', data=None ):
    if data:
        print("[dbg]", section, comment, data )
    else:
        print("[dbg]", section, comment  )

