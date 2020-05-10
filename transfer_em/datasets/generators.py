"""Contains some generator helper functions retrieving image data.
"""

def volume3d_dvid(dvid_server, uuid, instance, bbox, size=32):
    """Returns a generator that will produce an infinite number of 3D volumes
    from DVID.

    Note: only support uint8blk.
    """

    try:
        from libdvid import DVIDNodeService
    except ImportError:
        raise Exception("libdvid not installed")

    def generator():
        ns = DVIDNodeService(dvid_server, uuid)
        while True:
            #  get random starting point from bbox (x1,y1,z1) (x2,y2,z2)
            xstart = tf.random.unform(shape=[], minval=bbox[0][0], maxval=bbox[1][0], dtype=tf.int64)
            ystart = tf.random.unform(shape=[], minval=bbox[0][1], maxval=bbox[1][1], dtype=tf.int64)
            zstart = tf.random.unform(shape=[], minval=bbox[0][2], maxval=bbox[1][2], dtype=tf.int64)
            yield tf.convert_to_tensor(ns.get_gray3D(instance), (size, size, size),
                    (xstart, ystart, zstart), dtype=tf.uint8)

def volume3d_ng(location, size=32):
    """Returns a generator that will prodduce an infinite number of 3D volumes
    that are stored in neuroglancer format.

    TODO
    """
    
    try:
        import tensorstore as ts
    except ImportError:
        raise Exception("tensorstore not installed")

    # TODO 
    raise Exception("ng volume retrieval not implemented yet")


