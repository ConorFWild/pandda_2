import pathlib as p
import pandas as pd


def get_reference_map(reference_map_getter, reference, map_resolution, grid):
    ref_map = reference_map_getter(reference, map_resolution, grid)

    return ref_map


def fit(model, train, test):
    fit_model = model.fit(train, test)

    return fit_model


def load_sample(loader, dataset, grid, reference_map, map_resolution):
    xmap = loader(dataset, grid, reference_map, map_resolution)

    return xmap


def evaluate_model(model, xmap):
    zmap = model.evaluate(xmap)

    return zmap


def cluster_outliers(clusterer, dataset, zmap, grid):
    clusters = clusterer(dataset, zmap, grid)

    return clusters


def filter_clusters(cluster_filterer, dataset, clusters, grid):
    num_clusters, z_clusters = clusters
    events = cluster_filterer(dataset, num_clusters, z_clusters, grid)

    return events


def estimate_bdcs(bdc_estimator,
                  dataset,
                  dataset_map,
                  ref_map,
                  events,
                  grid,
                  ):
    bdcs = bdc_estimator(dataset,
                         dataset_map,
                         ref_map,
                         events,
                         grid,
                         )

    return bdcs


def make_event_map(map_maker, dataset_path, xmap, truncated_dataset, ref_map, event, bdc, statistical_model, grid):
    # dataset_path = p.Path(tree(("processed_datasets", truncated_dataset.tag))[0])
    # map_loader, truncated_dataset, ref_map, events, bdcs, dataset_path, grid, statistical_model
    # map_maker.process_single(map_loader, truncated_dataset, ref_map, events, bdcs, dataset_path, grid, statistical_model)
    map_maker(xmap,
              truncated_dataset,
              ref_map,
              event,
              bdc,
              dataset_path,
              statistical_model,
              grid,
              )


# def make_shell_maps(map_maker, tree, name, map_loader, ref_map, events, bdcs, grid, statistical_model, truncated_dataset):
def make_shell_maps(map_maker, mean_map_path, ref_dataset, ref_map):
    # Produce maps that are shared by iteration
    # dir_path = p.Path(tree([str(name)])[0])
    # dataset_path_string = str(dir_path / "mean_map.ccp4")
    # map_maker.process_shell(map_loader,
    #                              truncated_dataset,
    #                              ref_map,
    #                              events,
    #                              bdcs,
    #                              dir_path_string,
    #                              grid,
    #                              statistical_model)
    #
    # map_maker.process_shell(map_loader,
    #                         truncated_dataset,
    #                         ref_map,
    #                         events,
    #                         bdcs,
    #                         dir_path_string
    #                         )
    map_maker.process_shell(ref_dataset,
                            ref_map,
                            mean_map_path,
                            )


def make_event_table(event_table_maker,
                     event_table_path,
                     dataset,
                     events,
                     grid,
                     events_analysed,
                     ):
    # Produce the event table
    # dir_path = p.Path(tree([str(name)])[0])
    # event_table_path = dir_path / "event_table.csv"

    event_table = event_table_maker(dataset.partition_datasets("test"),
                                    events,
                                    event_table_path,
                                    grid,
                                    events_analysed,
                                    )

    return event_table


def merge_event_tables(event_tables):
    event_table = pd.concat([shell_table
                             for shell_num, shell_table
                             in event_tables.items()
                             ]
                            )
    return event_table


def output_event_table(event_table, event_table_path):
    event_table.to_csv(event_table_path)




# def criticise(criticiser, model, dataset, events):
#
#     event_table = criticiser(model, dataset, events)
#
#     return event_table
#
#
# def criticise_all(criticiser_all, model, events_list):
#
#     shell_event_table = criticiser_all(model, events_list)
#
#     return shell_event_table
