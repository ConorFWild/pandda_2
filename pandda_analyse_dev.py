import os, sys, glob, copy, time, gc, traceback, shutil, psutil, subprocess, re

# Regular Imports
import os, sys, configparser
import time

# Scientific imports
from joblib import Memory

import dask
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SGECluster
from dask import delayed

# Dask functions
from functions import fit, get_reference_map, load_sample, evaluate_model, cluster_outliers, filter_clusters, estimate_bdcs, make_event_map, make_shell_maps, make_event_table

# Multi Dataset Crystalography imports
import multi_dataset_crystalography as mdc # import MultiCrystalDataset
from multi_dataset_crystalography.utils import DefaultPanDDADataloader
from multi_dataset_crystalography.dataset.sample_loader import PanddaDiffractionDataTruncater

# PanDDA imports
from pandda_analyse.config import PanDDAConfig
from pandda_analyse.event_model import PanDDAEventModel
# from pandda_analyse.processor import ProcessModelSeriel
from pandda_analyse.event_model_distributed import PanDDAEventModelDistributed

from config import parse_phil_args
from functional import chain
from scheduler import get_client

#################################
try:
    import matplotlib

    print("Setting MPL backend")
    matplotlib.use('agg')
    matplotlib.interactive(False)
    print(matplotlib.get_backend())
    from matplotlib import pyplot

    pyplot.style.use('ggplot')
    print(pyplot.get_backend())
except Exception as e:
    print("Errored in mpl setup!")
    print(e)
#################################

# import pathlib as pth
#
# from bamboo.common import Meta
# from bamboo.common.logs import Log
# from bamboo.common.path import easy_directory
#
# from giant.jiffies import extract_params_default
#
#
# # from pandda import welcome, module_info
# from phil import pandda_phil

# from tasks.checkpointed_variable import CheckpointedVariable
# from tasks.pandda_args_processor import PanddaArgsProcessor
# from tasks.pandda_data_loader import PanddaDataLoader
# from tasks.pandda_data_preprocessor_subtasks.pandda_reference_structure_selecter import PanddaReferenceStructureSelecter
# from tasks.pandda_data_preprocessor_subtasks.pandda_data_checker import PanddaDataChecker
# from tasks.pandda_data_preprocessor_subtasks.pandda_diffraction_scaler import PanddaDiffractionScaler
# from tasks.pandda_data_preprocessor_subtasks.pandda_dataset_filterer import PanddaDatasetFilterer
# from tasks.pandda_data_preprocessor_subtasks.pandda_dataset_filterer_wilson_RMSD import PanddaDatasetFiltererWilsonRMSD
# from tasks.pandda_dataset_aligner import PanddaDatasetAligner
# from tasks.pandda_variable_collator import PanddaVariableCollator
# from tasks.pandda_resolution_shell_finder import PanddaResolutionShellFinder
# from tasks.pandda_default_select_characterisation_datasets import PanddaDefaultSelectCharacterisationDatasets
# from tasks.pandda_setup_recorder import PanddaSetupRecorder
# from tasks.pandda_default_analysis_dataset_selecter import PanddaDefaultAnalysisDatasetSelecter
# from tasks.pandda_main_loop import PanddaMainLoop
# from tasks.pandda_run_summariser import PanddaRunSummariser


class PanDDA:

    def __init__(self, args):
        self.args = args

    def __call__(self):

        pandda_start_time = time.time()

        working_phil = parse_phil_args(master_phil=pandda_phil,
                                       args=self.args,
                                       blank_arg_prepend=None,
                                       home_scope=None)

        # # TODO: remove
        # print("printing welcome")
        # sys.stdout.flush()
        # # welcome()
        #
        # # TODO: remove
        # print("getting phil")
        # sys.stdout.flush()
        # p = working_phil.extract()
        #
        # # TODO: remove
        # print("making directories")
        # sys.stdout.flush()
        # out_dir = easy_directory(os.path.abspath(p.pandda.output.out_dir))
        # _log_dir = easy_directory(os.path.join(out_dir, 'logs'))
        # # TODO: remove
        # print("made directories")
        # sys.stdout.flush()
        #
        # _log_filename = 'pandda-{}.log'.format(time.strftime("%Y-%m-%d-%H%M", time.gmtime()))
        #
        # # TODO: remove
        # print("got log fileename")
        # sys.stdout.flush()
        # _def_filename = _log_filename.replace('.log', '.def')
        # _eff_filename = _log_filename.replace('.log', '.eff')
        #
        # # TODO: remove
        # print("makeing log")
        # sys.stdout.flush()
        # log = Log(log_file=os.path.join(_log_dir, _log_filename), verbose=p.settings.verbose)
        #
        # # TODO: remove
        # print("args processor")
        # sys.stdout.flush()
        # pandda_arg_processor = PanddaArgsProcessor(log, p.pandda)
        # args = pandda_arg_processor()
        # params = args.params
        # settings = p.settings
        # # TODO: remove
        # print("got settings")
        # sys.stdout.flush()
        #
        # pickled_dataset_meta = Meta({'number_of_datasets': 0, 'dataset_labels': [], 'dataset_pickle_list': []})

        ################################################################################################################

        # Maps options to code abstractions
        pandda_config = PanDDAConfig(working_phil)

        # Get Dataset
        pandda_dataset = mdc.dataset.dataset.MultiCrystalDataset(dataloader=pandda_config.dataloader,
                                                                 sample_loader=pandda_config.sample_loader
                                                                 )

        reference = pandda_config.get_reference(pandda_dataset.datasets)
        pandda_dataset.sample_loader.reference = reference

        # transform dataset
        dataset = chain(pandda_dataset, [pandda_config.check_data,
                                         pandda_config.scale_diffraction,
                                         pandda_config.filter_structure,
                                         pandda_config.filter_wilson,
                                         pandda_config.align])

        # Get grid
        grid = pandda_config.get_grid(reference)

        # Get file tree
        tree = pandda_config.pandda_output(dataset)

        # Define event Model
        pandda_event_model = PanDDAEventModel(pandda_config.statistical_model,
                                              pandda_config.clusterer,
                                              pandda_config.event_finder,
                                              bdc_calculator=pandda_config.bdc_calculator,
                                              statistics=[],
                                              map_maker=pandda_config.map_maker,
                                              event_table_maker=pandda_config.event_table_maker,
                                              cpus=pandda_config["args"]["cpus"],
                                              tree=tree)

        # Get partitions
        partitions = pandda_config.partitioner(dataset)

        # instatiate a dataloader for the datasets
        # dataloader = DefaultPanDDADataloader(min_train_datasets=60,
        #                                      max_test_datasets=60)

        # Get the datasets to iterate over
        ds = [(idx, d) for idx, d in pandda_config.dataloader(dataset)]

        # Iterate over resolution shells
        for shell_num, shell_dataset in ds:

            client = get_client()

            # ###############################################
            # Get resolution
            # ###############################################
            resolutions_test = max([dts.data.summary.high_res for dtag, dts
                                    in shell_dataset.partition_datasets("test").items()])
            resolutions_train = max([dts.data.summary.high_res for dtag, dts
                                     in shell_dataset.partition_datasets("train").items()])
            max_res = max(resolutions_test, resolutions_train)

            # ###############################################
            # Instantiate sheel variable names
            # ###############################################

            # Dataset names
            dtags = set(shell_dataset.partition_datasets("test").keys()
                        + shell_dataset.partition_datasets("train").keys()
                        )

            dask_dtags = {"{}".format(dtag.replace("-", "_")): dtag
                          for dtag
                          in dtags}
            train_dtags = [dtag
                           for dtag
                           in dask_dtags
                           if (dask_dtags[dtag] in shell_dataset.partition_datasets("train").keys())]
            test_dtags = [dtag
                          for dtag
                          in dask_dtags
                          if (dask_dtags[dtag] in shell_dataset.partition_datasets("test").keys())]

            # ###############################################
            # Truncate datasets
            # ###############################################
            # TODO: move to imports section

            truncated_reference, truncated_datasets = PanddaDiffractionDataTruncater()(shell_dataset.datasets,
                                                                                       reference)

            # ###############################################
            # Load computed variables into dask
            # ###############################################
            # Rename trucnated datasets
            for ddtag, dtag in dask_dtags.items():
                truncated_datasets[ddtag] = truncated_datasets[dtag]

            # record max res of shell datasets
            shell_max_res = max_res

            # ###############################################
            # Generate maps
            # ###############################################
            # Generate reference map for shell
            shell_ref_map = delayed(get_reference_map)(pandda_config.reference_map_getter,
                                                       reference,
                                                       shell_max_res,
                                                       grid)

            # Load maps
            xmaps = {}
            for dtag in dask_dtags:
                xmaps[dtag] = delayed(load_sample)(pandda_config.map_loader,
                                                   truncated_datasets[dtag],
                                                   grid,
                                                   shell_ref_map,
                                                   shell_max_res)

            # ###############################################
            # Fit statistical model to trianing sets
            # ###############################################
            xmaps_persisted_futures = client.persist([xmaps[dtag] for dtag in dask_dtags])
            xmaps_computed = {dtag: client.compute(xmaps_persisted_futures[i]).result()
                              for i, dtag
                              in enumerate(dask_dtags)}


            shell_fit_model = fit(pandda_config.statistical_model,
                                  [xmaps_computed[dtag] for dtag in train_dtags],
                                  [xmaps_computed[dtag] for dtag in test_dtags]
                                  )

            shell_fit_model_scattered = client.scatter(shell_fit_model)

            xmaps_scattered = client.scatter([xmaps_computed[dtag] for dtag in dask_dtags])
            xmaps_scattered_dict = {dtag: xmaps_scattered[i] for i, dtag in enumerate(dask_dtags)}

            grid_scattered = client.scatter(grid)

            # ###############################################
            # Find events
            # ###############################################
            zmaps = {}
            clusters = {}
            events = {}
            bdcs = {}
            for dtag in dask_dtags:
                # Get z maps by evaluating model on maps
                zmaps[dtag] = delayed(evaluate_model)(shell_fit_model_scattered,
                                                      xmaps_scattered_dict[dtag]
                                                      )

                # Cluster outlying points in z maps
                clusters[dtag] = delayed(cluster_outliers)(pandda_config.clusterer,
                                                           truncated_datasets[dtag],
                                                           zmaps[dtag],
                                                           grid_scattered
                                                           )

                # Find events by filtering the clusters
                events[dtag] = delayed(filter_clusters)(pandda_config.event_finder,
                                                        truncated_datasets[dtag],
                                                        clusters[dtag],
                                                        grid_scattered
                                                        )

            events_persisted_futures = client.persist([events[dtag] for dtag in dask_dtags])
            events_computed = {dtag: client.compute(events_persisted_futures[i]).result()
                               for i, dtag
                               in enumerate(dask_dtags)}

            events_scattered = client.scatter([events_computed[dtag] for dtag in dask_dtags])
            events_scattered_dict = {dtag: xmaps_scattered[i] for i, dtag in enumerate(dask_dtags)}

            # Calculate background correction factors
            for dtag in dask_dtags:
                bdcs[dtag] = delayed(estimate_bdcs)(pandda_config.bdc_calculator,
                                                    truncated_datasets[dtag],
                                                    xmaps_scattered_dict[dtag],
                                                    shell_ref_map,
                                                    events[dtag],
                                                    grid_scattered
                                                    )

            # Criticise each indiidual dataset (generate statistics, event map and event table)
            event_maps = {}
            for dtag in dask_dtags:
                event_maps[dtag] = delayed(make_event_map)(pandda_config.map_maker,
                                                           tree,
                                                           pandda_config.map_loader,
                                                           truncated_datasets[dtag],
                                                           shell_ref_map,
                                                           events[dtag],
                                                           bdcs[dtag])

            event_maps_persisted_futures = client.persist([event_maps[dtag] for dtag in dask_dtags])

            event_maps_computed = {dtag: client.compute(event_maps_persisted_futures[i]).result()
                                   for i, dtag
                                   in enumerate(dask_dtags)}

            shell_maps = delayed(make_shell_maps)(pandda_config.map_maker,
                                                  tree,
                                                  shell_num,
                                                  reference,
                                                  shell_ref_map
                                                  )

            shell_maps_persisted_futures = client.persist(shell_maps)
            shell_maps_computed = shell_maps_persisted_futures.result()

            event_table = delayed(make_event_table)(pandda_config.event_table_maker,
                                                    tree,
                                                    shell_num,
                                                    shell_dataset,
                                                    events_computed
                                                    )

            event_table_persisted_future = client.persist(event_table)
            event_table_computed = event_table_persisted_future.result()

            client.close()



        # # ============================================================================>
        # # Flag running
        # # ============================================================================>
        # f = open(out_dir + "/pandda.running", "w")
        # f.write("\n")
        # f.close()
        #
        # # ============================================================================>
        # # Define Checkpointed Variables
        # # ============================================================================>
        # args_checkpoint = CheckpointedVariable(out_dir, "args")
        # args_checkpoint.val = args
        # settings_checkpoint = CheckpointedVariable(out_dir, "settings")
        # settings_checkpoint.val = settings
        # params_checkpoint = CheckpointedVariable(out_dir, "params")
        # params_checkpoint.val = params
        # settings_checkpoint = CheckpointedVariable(out_dir, "settings")
        # settings_checkpoint.val = settings
        #
        # pandda_datasets = CheckpointedVariable(out_dir, "pandda_datasets")
        # file_manager = CheckpointedVariable(out_dir, "file_manager")
        # pandda_pickle_manager = CheckpointedVariable(out_dir, "pandda_pickle_manager")
        # checked_datasets = CheckpointedVariable(out_dir, "checked_datasets")
        # reference_dataset = CheckpointedVariable(out_dir, "reference_dataset")
        #
        # scaled_datasets = CheckpointedVariable(out_dir, "scaled_datasets")
        # dataset_tables_w_scale = CheckpointedVariable(out_dir, "dataset_tables_w_scale")
        #
        # filtered_datasets = CheckpointedVariable(out_dir, "filtered_datasets")
        #
        # alignments = CheckpointedVariable(out_dir, "alignments")
        # grid = CheckpointedVariable(out_dir, "grid")
        # aligned_datasets = CheckpointedVariable(out_dir, "aligned_datasets")
        # rejected_dataset_tags = CheckpointedVariable(out_dir, "rejected_dataset_tags")
        #
        # characterisation_datasets = CheckpointedVariable(out_dir, "characterisation_datasets")
        #
        # not_excluded_dataset_tags = CheckpointedVariable(out_dir, "not_excluded_dataset_tags")
        #
        # resolution_shells = CheckpointedVariable(out_dir, "resolution_shells")
        #
        # dataset_tables_collated = CheckpointedVariable(out_dir, "dataset_tables_collated")
        #
        # characterisation_dataset_tags = CheckpointedVariable(out_dir, "characterisation_dataset_tags")
        #
        # analysis_dataset_tags = CheckpointedVariable(out_dir, "analysis_dataset_tags")
        #
        # setup_recorded = CheckpointedVariable(out_dir, "setup_recorded")
        #
        # # ============================================================================>
        # # Load Data
        # # ============================================================================>
        # # TODO: remove
        # print("loading data")
        # sys.stdout.flush()
        # if not ((pandda_datasets.iscached()) & (file_manager.iscached()) & (pandda_pickle_manager.iscached())):
        #     pandda_data_loader = PanddaDataLoader(log,
        #                                           args,
        #                                           pickled_dataset_meta,
        #                                           _def_filename,
        #                                           _eff_filename,
        #                                           out_dir)
        #     pandda_datasets.val, file_manager.val, pandda_pickle_manager.val = pandda_data_loader()
        #
        # # ============================================================================>
        # # Preprocess data
        # # ============================================================================>
        # # TODO: remove
        # print("preprocessing")
        # sys.stdout.flush()
        #
        # # TODO: these need to be memory fused as structure factors cannot be loaded from a pickle of an mtz file object
        # if not (checked_datasets.iscached()):
        #     pandda_data_checker = PanddaDataChecker(log, params,
        #                                             pandda_datasets.val)
        #     checked_datasets.val = pandda_data_checker()
        #
        # if not (reference_dataset.iscached()):
        #     pandda_reference_selecter = PanddaReferenceStructureSelecter(out_dir, log, args,
        #                                                                  pandda_datasets.val,
        #                                                                  file_manager.val)
        #     reference_dataset.val = pandda_reference_selecter()
        #
        # if not (scaled_datasets.iscached() & dataset_tables_w_scale.iscached()):
        #     pandda_diffraction_scaler = PanddaDiffractionScaler(log, params,
        #                                                         checked_datasets.val,
        #                                                         reference_dataset.val)
        #     scaled_datasets.val, dataset_tables_w_scale.val = pandda_diffraction_scaler()
        #
        # # ============================================================================>
        # # Initial filtering of the datasets on Structural deviation and wilson RMSD
        # # ============================================================================>
        # # TODO: remove
        # print("filtering")
        # sys.stdout.flush()
        # if not (filtered_datasets.iscached()):
        #     pandda_dataset_filterer = PanddaDatasetFilterer(log,
        #                                                     params,
        #                                                     scaled_datasets.val,
        #                                                     reference_dataset.val)
        #     filtered_datasets.val = pandda_dataset_filterer.filter_structural_deviation()
        #
        # if not (not_excluded_dataset_tags.iscached()):
        #     pandda_dataset_filterer = PanddaDatasetFiltererWilsonRMSD(log,
        #                                                               params,
        #                                                               filtered_datasets.val,
        #                                                               dataset_tables_w_scale.val)
        #     not_excluded_dataset_tags.val = pandda_dataset_filterer.filter_wilson_RMSD()
        #
        # # TODO: remove
        # print("gettting un-excvluded datasetsd")
        # sys.stdout.flush()
        #
        # not_excluded_datasets = [dataset
        #                          for dataset
        #                          in filtered_datasets.val
        #                          if (dataset.tag in not_excluded_dataset_tags.val)]
        #
        # # ============================================================================>
        # # Align datasets
        # # ============================================================================>
        # # TODO: remove
        # print("aligning")
        # sys.stdout.flush()
        # gc.collect()
        #
        # for dataset in filtered_datasets.val:
        #     dataset.model.alignment = None
        #
        # if not (alignments.iscached() & grid.iscached()):
        #     pandda_dataset_aligned = PanddaDatasetAligner(log,
        #                                                   args,
        #                                                   settings,
        #                                                   file_manager.val,
        #                                                   filtered_datasets.val,
        #                                                   reference_dataset.val)
        #     alignments.val, grid.val = pandda_dataset_aligned()
        #
        # alignments_hash = {alignment.id: alignment
        #                    for alignment
        #                    in alignments.val}
        #
        # if not aligned_datasets.iscached():
        #     aligned_datasets_tmp = []
        #     for dataset in filtered_datasets.val:
        #         ds_tmp = dataset
        #         ds_tmp.model.alignment = alignments_hash[ds_tmp.tag]
        #         aligned_datasets_tmp.append(ds_tmp)
        #     aligned_datasets.val = aligned_datasets_tmp
        #
        # # ============================================================================>
        # # Collate Variables
        # # ============================================================================>
        # # TODO: remove
        # print("collating")
        # sys.stdout.flush()
        # if not (dataset_tables_collated.iscached()):
        #     pandda_variable_collator = PanddaVariableCollator(log,
        #                                                       dataset_tables_w_scale.val,
        #                                                       aligned_datasets.val)
        #     dataset_tables_collated.val = pandda_variable_collator()
        #
        # # ============================================================================>
        # # Align datasets
        # # ============================================================================>
        # # TODO: remove
        # print("aligning")
        # sys.stdout.flush()
        # if not (resolution_shells.iscached()):
        #     pandda_resolution_shell_finder = PanddaResolutionShellFinder(log,
        #                                                                  args,
        #                                                                  params,
        #                                                                  dataset_tables_collated.val,
        #                                                                  reference_dataset.val)
        #     resolution_shells.val = pandda_resolution_shell_finder()
        # resolution_shell_recorders = {}
        #
        # resolution_lower_bound = 0
        #
        # res_limits = []
        #
        # for resolution_shell in resolution_shells.val:
        #     bounds = {"large": resolution_shell,
        #               "small": resolution_lower_bound}
        #     res_limits.append(bounds)
        #     resolution_lower_bound = resolution_shell
        #
        # # ============================================================================>
        # # Select the characterisation datasets
        # # ============================================================================>
        #
        # if not (characterisation_datasets.iscached()):
        #
        #     if args.input.flags.ground_state_datasets is not None:
        #
        #         print("Ground state datasets")
        #         print(args.input.flags.ground_state_datasets)
        #         print("Number of non-excluded datasets is: {}".format(len(not_excluded_datasets)))
        #
        #         possible_characterisation_dataset_tags = args.input.flags.ground_state_datasets.split(",")
        #
        #         possible_characterisation_datasets = [dataset
        #                                               for dataset
        #                                               in not_excluded_datasets
        #                                               if dataset.tag in possible_characterisation_dataset_tags]
        #         print("Number of possible_characterisation_dataset_tags: {}".format(
        #             len(possible_characterisation_dataset_tags)))
        #         print(
        #             "Number of possible_characterisation_datasets: {}".format(len(possible_characterisation_datasets)))
        #
        #         pandda_default_select_characterisation_datasets = PanddaDefaultSelectCharacterisationDatasets(log,
        #                                                                                                       params,
        #                                                                                                       dataset_tables_collated.val,
        #                                                                                                       possible_characterisation_datasets,
        #                                                                                                       res_limits)
        #         characterisation_dataset_tags.val = pandda_default_select_characterisation_datasets()
        #
        #     elif args.params.pre_pandda.do_pre_pandda:
        #         pandda_characterisation_dataset_selecter = PanddaCharacterisationDatasetSelecter()
        #         characterisation_datasets = PanddaCharacterisationDatasetSelecter(not_excluded_datasets,
        #                                                                           alignments,
        #                                                                           grid,
        #                                                                           scaled_mtzs)
        #     else:
        #         pandda_default_select_characterisation_datasets = PanddaDefaultSelectCharacterisationDatasets(log,
        #                                                                                                       params,
        #                                                                                                       dataset_tables_collated.val,
        #                                                                                                       not_excluded_datasets,
        #                                                                                                       res_limits)
        #         characterisation_dataset_tags.val = pandda_default_select_characterisation_datasets()
        #
        # # ============================================================================>
        # # Select the analysis datasets
        # # ============================================================================>
        #
        # # TODO: make sure that this takes the number of characterisation datasets into account so that datasets not assigned to non-running shells
        # analysis_dataset_tags_tmp = PanddaDefaultAnalysisDatasetSelecter(log,
        #                                                                  dataset_tables_collated.val,
        #                                                                  aligned_datasets.val,
        #                                                                  res_limits)()
        # dataset_tag_acculator = []
        # for res_shell in resolution_shells.val:
        #     if characterisation_dataset_tags.val[res_shell] is None:
        #         dataset_tag_acculator = dataset_tag_acculator + analysis_dataset_tags_tmp[res_shell]
        #     else:
        #         if len(dataset_tag_acculator) != 0:
        #             analysis_dataset_tags_tmp[res_shell] = analysis_dataset_tags_tmp[res_shell] + dataset_tag_acculator
        #         break
        #
        # print("Adding the following datasets to resolution shell {}: \n{}".format(res_shell,
        #                                                                           dataset_tag_acculator))
        # analysis_dataset_tags.val = analysis_dataset_tags_tmp
        #
        # # ============================================================================>
        # # Record and output a summary of the setup stages
        # # ============================================================================>
        # accepted_dataset_tags = [dataset.tag
        #                          for dataset
        #                          in aligned_datasets.val]
        # rejected_dataset_tags.val = [dataset.tag
        #                              for dataset
        #                              in pandda_datasets.val.all()
        #                              if not (dataset.tag in accepted_dataset_tags)]
        # rejected_datasets = [dataset
        #                      for dataset
        #                      in aligned_datasets.val
        #                      if (dataset.tag in rejected_dataset_tags.val)]
        #
        # if not (setup_recorded.iscached()):
        #     pandda_setup_recorder = PanddaSetupRecorder(args,
        #                                                 log,
        #                                                 dataset_tables_collated.val,
        #                                                 file_manager.val,
        #                                                 aligned_datasets.val,
        #                                                 resolution_shells.val,
        #                                                 rejected_datasets,
        #                                                 reference_dataset.val
        #                                                 )
        #     setup_recorded.val = pandda_setup_recorder()
        #
        # # ============================================================================>
        # # Main PanDDA loop
        # # ============================================================================>
        # del pandda_datasets
        # del checked_datasets
        # del scaled_datasets
        # del characterisation_datasets
        # del filtered_datasets
        # del alignments_hash
        # del setup_recorded
        # gc.collect()
        #
        # t_anal_start = time.time()
        #
        # # TODO: set this more sensibly
        # distributed = True
        #
        # if distributed is True:
        #
        #     # TODO
        #     print("Submitting jobs!")
        #     sys.stdout.flush()
        #     for resolution_shell in resolution_shells.val:
        #         f = open("submit_script_{}.sh".format(float(resolution_shell)), "w")
        #         # f.write(
        #         #     "chmod 777 /dls/science/groups/i04-1/conor_dev/pandda/lib-python/pandda/analyse/tasks/pandda_main_loop.py\n")
        #         f.write(
        #             "/dls/science/groups/i04-1/conor_dev/ccp4/build/bin/cctbx.python /dls/science/groups/i04-1/conor_dev/pandda/lib-python/pandda/analyse/tasks/pandda_main_loop.py {} {} {}".format(
        #                 out_dir,
        #                 float(resolution_shell),
        #                 t_anal_start))
        #         f.close()
        #         p = subprocess.Popen("chmod 777 submit_script_{}.sh".format(float(resolution_shell)),
        #                              shell=True)
        #         p.communicate()
        #
        #         submit_name = "pandda_" + "{}".format(float(resolution_shell))
        #
        #         command = "qsub -P labxchem -q medium.q -N {} -l exclusive,m_mem_free=64G submit_script_{}.sh".format(
        #             submit_name,
        #             float(resolution_shell))
        #         p = subprocess.Popen(command,
        #                              shell=True)
        #         p.communicate()
        #         time.sleep(1.50)
        #
        #     # ============================================================================>
        #     # Check all completed
        #     # ============================================================================>
        #     print(resolution_shells.val)
        #     running = True
        #     while running is True:
        #         print("Checking jobs!")
        #         sys.stdout.flush()
        #         flag_paths = {res_shell: pth.Path(out_dir) / str(res_shell) / "done.txt"
        #                       for res_shell
        #                       in resolution_shells.val}
        #         print([path for path in flag_paths])
        #
        #         process_states = {p: flag_paths[p].exists()
        #                           for p
        #                           in flag_paths}
        #         print([path for path in process_states])
        #         print([process_states[p] for p in process_states])
        #         finished_tasks = sorted([p for p in process_states if process_states[p]])
        #         unfinished_tasks = sorted([p for p in process_states if not process_states[p]])
        #         print("Done: {}".format(finished_tasks))
        #         print("Still going: {}".format(unfinished_tasks))
        #
        #         if len(unfinished_tasks) == 0:
        #             break
        #
        #         # =============================
        #         # Resubmit any failed jobs
        #         # =============================
        #         p = subprocess.Popen("qstat -r",
        #                              shell=True,
        #                              stdout=subprocess.PIPE)
        #         job_stats_string = p.communicate()[0]
        #
        #         for resolution_shell in resolution_shells.val:
        #             submit_name = "pandda_" + "{}".format(float(resolution_shell))
        #             submit_name = submit_name.replace(".", "\\.")
        #             if resolution_shell in unfinished_tasks:
        #
        #                 print("Shell {} is unfinished: checking if failed".format(resolution_shell))
        #                 print("Matching pattern - {} - in job stats string".format(submit_name))
        #                 print(re.match(submit_name, job_stats_string))
        #                 if re.search(submit_name, job_stats_string) is None:
        #                     print("Shell {} seems to have failed - resubmitting".format(resolution_shell))
        #                     command = "qsub -P labxchem -q medium.q -N {} -l exclusive,m_mem_free=64G submit_script_{}.sh".format(
        #                         submit_name,
        #                         float(resolution_shell))
        #                     p = subprocess.Popen(command,
        #                                          shell=True)
        #                     p.communicate()
        #                     time.sleep(1.50)
        #
        #         time.sleep(60.0)
        #
        #
        #
        # else:
        #     for resolution_shell in resolution_shells.val:
        #         log.heading("Working on resolution shell {}".format(resolution_shell))
        #         main_loop = PanddaMainLoop(out_dir, resolution_shell, log, args, params, settings, t_anal_start)
        #         main_loop()
        #
        # # ============================================================================>
        # # Clean
        # # ============================================================================>
        # for resolution_shell in resolution_shells.val:
        #     command = "rm submit_script_{}.sh".format(float(resolution_shell))
        #     p = subprocess.Popen(command,
        #                          shell=True)
        #     p.communicate()
        #
        # # ==============================>
        # # Load the summary table
        # # ==============================>
        #
        # resolution_shell_paths = {r: pth.Path(out_dir) / "{}".format(r)
        #                           for r
        #                           in resolution_shells.val}
        #
        # tables_dic = {
        #     r: (resolution_shell_paths[r] / "shell_table" if (resolution_shell_paths[r] / "shell_table").exists()
        #         else None)
        #     for r
        #     in resolution_shells.val}
        #
        # event_table = None
        # for key, table_path in tables_dic.items():
        #     if table_path is not None:
        #
        #         tables = easy_pickle.load(str(table_path))
        #
        #         if event_table is None:
        #             print("setting event table at {}".format(key))
        #
        #             event_table = tables.event_info
        #         else:
        #             print("updating event table at {}".format(key))
        #             #             event_table.update(tables.event_info)
        #             #             event_table = event_table.merge(tables.event_info, on=["dtag", "event_idx"], how="outer")
        #             event_table = pd.concat([event_table, tables.event_info])
        #
        # print(event_table.sort_values("dtag"))
        # print(len(event_table))
        #
        # dataset_map_info_table = None
        # for key, table_path in tables_dic.items():
        #     if table_path is not None:
        #
        #         tables = easy_pickle.load(str(table_path))
        #
        #         if event_table is None:
        #             print("setting event table at {}".format(key))
        #             analysed_map_info = tables.dataset_map_info[tables.dataset_map_info["analysed_resolution"] > 0]
        #             dataset_map_info_table = analysed_map_info
        #         else:
        #             print("updating event table at {}".format(key))
        #             analysed_map_info = tables.dataset_map_info[tables.dataset_map_info["analysed_resolution"] > 0]
        #
        #             dataset_map_info_table = pd.concat([dataset_map_info_table, analysed_map_info])
        #
        # tables = dataset_tables_collated.val
        # tables.event_info = event_table
        # tables.dataset_map_info = dataset_map_info_table
        #
        # # ==============================>
        # # Output a summary information on the PanDDA run
        # # ==============================>
        # pandda_run_summariser = PanddaRunSummariser(log,
        #                                             tables,
        #                                             file_manager.val,
        #                                             args,
        #                                             grid.val,
        #                                             analysis_dataset_tags.val,
        #                                             characterisation_dataset_tags.val,
        #                                             rejected_dataset_tags.val,
        #                                             reference_dataset.val)
        # # TODO: make sure this only takes analysed datasets or filters them internally so htmls are only written for those
        # pandda_run_summariser()
        #
        # pandda_finish_time = time.time()
        #
        # print("PanDDA ran in {}".format(pandda_finish_time - pandda_start_time))
        #
        # # ============================================================================>
        # # Flag finished
        # # ============================================================================>
        # f = open(out_dir + "/pandda.done", "w")
        # f.write("\n")
        # f.close()


if __name__ == "__main__":
    pandda = PanDDA(sys.argv[1:])
    pandda()







