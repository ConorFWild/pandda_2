import time

import dask
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SGECluster


def get_client():

    dask.config.set({"distributed.admin.tick.limit": "300s"})

    cluster = SGECluster(queue="medium.q",
                         project="labxchem",
                         cores=10,
                         processes=5,
                         memory="64GB",
                         resource_spec="m_mem_free=64G,redhat_release=rhel7",
                         python="/dls/science/groups/i04-1/conor_dev/ccp4/build/bin/cctbx.python",
                         walltime="03:00:00",
                         )
    cluster.scale(60)

    time.sleep(15)

    client = Client(cluster)

    return client


def replicate_dict_futures(client, futures_dict, n):
    client.replicate([future
                      for key, future
                      in futures_dict.items()
                      ],
                     n=n
                     )