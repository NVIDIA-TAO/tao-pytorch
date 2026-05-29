# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Daatset Preprocess - Integrates into Factory camera system pipeline"""
import pandas as pd
import os
import numpy as np
import shutil
from nvidia_tao_pytorch.core.tlt_logging import logging


def get_boards_perproject(df, project_name, top_bot, compare_csv_path):
    """Get Golden boards per project

    Args:
        df (pd.DataFrame): The input DataFrame containing board projects information.
        top_bot (str): The top/bottom configuration.
        compare_csv_path (str): The path to the compare CSV files.
        project_name (str): The name of the project to get information for.

    """
    files_project = df[(df['project'] == project_name) & (df['TB'] == top_bot)]['Files']
    project_boards_csv = pd.DataFrame()
    for fname in files_project:
        train0 = pd.read_csv(compare_csv_path + fname)
        project_boards_csv = pd.concat([train0, project_boards_csv], axis=0)
    project_boards_csv = project_boards_csv.reset_index()

    return files_project, project_boards_csv


def copy_golden_images(comparefile, proj_name, top_bot, golden_main_path, image_root, goldencsv_fname):
    """Copy golden images for a specific project and top/bottom configuration.

    Args:
        comparefile (pd.DataFrame): The DataFrame containing the project information.
        proj_name (str): The name of the project.
        top_bot (str): The top/bottom configuration.
        golden_main_path (str): The path to the golden main directory to save the golden images at.
        image_root (str): The root path to the image directory.
        goldencsv_fname (str): The file name for the output golden CSV.

    """
    golden_path = '/'.join([golden_main_path, 'images', proj_name + top_bot, ''])
    if not os.path.exists(golden_path):
        os.mkdir(golden_path)
    lighting_conditions = list(comparefile['LightingCondition'].unique())

    cnt_df = comparefile.groupby(
        ['CompName', 'TB_Panel', 'BoardSN']).size().reset_index(name='counts')
    comparefile = pd.merge(
        comparefile, cnt_df, how='left', on=['CompName', 'TB_Panel', 'BoardSN'])
    comparefile = comparefile.loc[comparefile['counts'] == 4]

    comparefile = comparefile.loc[(comparefile['MachineDefect'] == 'PASS') &
                                  (comparefile['ConfirmDefect'] == 'PASS')]

    goldenclean = comparefile.drop_duplicates(
        subset=['CompName', 'TB_Panel'], keep="last").reset_index()
    goldenclean = goldenclean[['CompName', 'directory', 'TB_Panel', 'BoardSN', 'Project', 'TOP_BOT']]

    for row in range(0, goldenclean.shape[0], 1):
        records = goldenclean.iloc[row, :]
        image_name = records['CompName'] + '@' + str(records['TB_Panel'])
        image_loc = image_root + records['directory']
        for light in lighting_conditions:
            img_light_path = image_loc + '/' + image_name + '_' + light + '.jpg'
            shutil.copy(img_light_path, golden_path)

    goldenclean.rename({'boardSN': 'boardSN_used'}, axis=1, inplace=True)
    goldenclean['directory'] = golden_path
    goldenclean.to_csv(goldencsv_fname, index=False, encoding='utf-8', header=True)


def create_golden_forprojects(df, root_data_path, golden_csv_path, compare_csv_path, project_list):
    """Create golden paths for multiple projects.

    Args:
        df (pd.DataFrame): The input DataFrame containing board information.
        root_data_path (str): The root path to the data directory.
        golden_csv_path (str): The path where the golden CSV files will be saved.
        compare_csv_path (str): The path to the compare CSV files.
        project_list (list): A list of projects to copy golden images for.
    """
    projects_list = project_list
    for proj_info in projects_list:
        project_name = proj_info.name
        top_bot = proj_info.top_bot
        _, csvfiles_concat = get_boards_perproject(df, project_name, top_bot, compare_csv_path)
        goldencsv_fname = golden_csv_path + project_name + top_bot + '.csv'

        if os.path.exists(goldencsv_fname):
            continue
        logging.info('creating Golden for {}{}'.format(project_name, top_bot))
        golden_main_path = root_data_path + 'dlout/'
        image_root = root_data_path + 'images/'
        copy_golden_images(csvfiles_concat, project_name, top_bot, golden_main_path, image_root, goldencsv_fname)


def preprocess_boards_cam(summaryfiles_path, top_bot):
    """Pre-process boards"""
    def get_top_bot(x):
        # print(x)
        if x.find('~') != -1:
            return x.split('_')[1].split('~')[1]

        return x.split('_')[2]

    def get_boardsn(x):
        # print(x)
        if x.find('~') != -1:
            return x.split('_')[2]

        return x.split('_')[3]

    df = pd.DataFrame(os.listdir(summaryfiles_path), columns=['Files'])
    df = df.loc[df['Files'].str.endswith('.csv')]
    df['project'] = df['Files'].apply(lambda x: x.split('_')[1].split('~')[0])
    df['boardname'] = df['Files'].apply(lambda x: get_boardsn(x))
    df['TB'] = df['Files'].apply(lambda x: get_top_bot(x))

    if top_bot != 'all':
        if (top_bot[0] in ['TOP', 'BOT']):
            df = df.loc[(df['TB'].isin(top_bot))]
        else:
            logging.error("INPUT VALID VALUE FOR top_bot")

    return df


def aggr(ser_):
    """Checking PASS FAIL files and aggregating"""
    allpass = all([x == 'PASS' for x in ser_])
    if allpass:
        return 'PASS'
    return '_'.join(np.unique([x for x in ser_ if x != 'PASS']))


def get_top_bot(topbot):
    """Get Top Bot"""
    if topbot == 'BOT':
        return 'AOI_B'

    return 'AOI_T'


def aggr_status_across_lights(df):
    """Aggregating status across lights"""
    comp_board_cnst_cols = ['CompName', 'directory']
    for col in ['ConfirmDefect', 'MachineDefect', 'Status']:
        combined = df.groupby(comp_board_cnst_cols)[col].apply(aggr).reset_index()
        df = df.drop(col, axis=1)
        df = pd.merge(df, combined, how='left', on=comp_board_cnst_cols)
    return df


def get_allcomp_csv(df_, compare_csv_path, data_path):
    """Getting all components CSV

    Args:
        df_ (pd.DataFrame): The input DataFrame containing the project information.
        data_path (str): The path to the data directory.
        compare_csv_path (str): The path to the CSV file containing compare images.
    """
    files = df_['Files']
    match_csv = pd.DataFrame()
    for fname in files:
        train0 = pd.read_csv(data_path + compare_csv_path + fname)

        checkprojectpath = True
        if checkprojectpath:
            dir_tuple = train0.iloc[0, :]
            project = dir_tuple['Project']
            top_bot = dir_tuple['TOP_BOT']
            if dir_tuple.directory.split('/')[0] != project:
                train0['directory'] = project + '/' + get_top_bot(top_bot) + '/' + dir_tuple['directory']

        match_csv = pd.concat([train0, match_csv], axis=0)

    comp_on_board_cols = ['Project', 'BoardSN', 'TOP_BOT', 'CompName', 'TB_Panel', 'directory']
    cnt = match_csv.groupby(comp_on_board_cols).size().reset_index(name='light_count')
    match_csv = pd.merge(match_csv, cnt, how='left', on=comp_on_board_cols)
    num_img = match_csv.shape[0]
    match_csv = match_csv.loc[match_csv['light_count'] == 4]
    print('Out of {}, dropped {} images due to != 4 lighting conditions'.format(
        num_img, num_img - match_csv.shape[0]))

    false_call_idx = match_csv['ConfirmDefect'].isna()
    print("In ConfirmDefect, for {} rows, NaN replaced with PASS".format(
        match_csv[false_call_idx].shape[0]))
    match_csv.loc[false_call_idx, 'ConfirmDefect'] = 'PASS'
    return match_csv


def move_golden(data_path, goldencsv, savemaplight):
    """Moving golden images to images directory

    Args:
        data_path (str): The path to the data directory.
        goldencsv (str): The golden CSV dataframe.
        savemaplight (str): Flag indicating whether {SaveMapLight} is present in
                    original image paths before pre-processing the dataset paths.
    """
    golden_path = data_path + 'images/golden'
    if not os.path.exists(golden_path):
        os.mkdir(golden_path)
    golden_path = data_path + 'images/golden/images'
    if not os.path.exists(golden_path):
        os.mkdir(golden_path)
    golden_path = data_path + 'images/' + goldencsv.directory[0]
    if not os.path.exists(golden_path):
        os.mkdir(golden_path)

    for row in range(0, goldencsv.shape[0], 1):
        records = goldencsv.iloc[row, :]
        image_name = records['CompName'] + '@' + str(records['TB_Panel'])
        pre_golden_path = data_path + 'dlout/' + records['directory']
        post_golden_path = data_path + 'images/' + records['directory']
        for light in ['LowAngleLight', 'SolderLight', 'UniformLight', 'WhiteLight']:
            if savemaplight:
                pre_img_light_path = pre_golden_path + image_name + '{SaveMapLight}' + '_' + light + '.jpg'
                post_img_light_path = post_golden_path + image_name + '_' + light + '.jpg'
            else:
                pre_img_light_path = pre_golden_path + image_name + '_' + light + '.jpg'
                post_img_light_path = post_golden_path + image_name + '_' + light + '.jpg'
            shutil.copy(pre_img_light_path, post_img_light_path)


def getgoldenpaths(goldencsv):
    """Get golden paths"""
    listG = goldencsv.directory[0].split('/')
    listG = listG[listG.index('golden'):]
    gpath = '/'.join(listG)
    return gpath


def getgolden(project_name, top_bot, golden_csv_path, data_path, movegoldenimgs, savemaplight):
    """Getting Golden Boards for a specific project and top/bottom configuration.

    Args:
        project_name (str): The name of the project.
        top_bot (str): The top/bottom configuration.
        data_path (str): The path to the data directory.
        golden_csv_path (str): The path to the golden CSV file.
        movegoldenimgs (bool): Flag indicating whether to move golden images to images directory.
        savemaplight (str): Flag indicating whether {SaveMapLight} is present in
                    original image paths before pre-processing the dataset paths.

    Returns:
        pandas.DataFrame: A DataFrame containing the golden board information.
    """
    goldencsv = pd.read_csv(data_path + golden_csv_path + project_name + top_bot + '.csv')

    if goldencsv.directory[0].split('/')[0] == 'golden':
        if movegoldenimgs:
            move_golden(data_path, goldencsv, savemaplight)

        return goldencsv

    goldencsv['directory'] = getgoldenpaths(goldencsv)

    if movegoldenimgs:
        move_golden(data_path, goldencsv, savemaplight)

    return goldencsv


def getboards_mergedw_golden(df_, golden_csv_path, compare_csv_path, data_path, movegoldenimgs, savemaplight):
    """Getting boards merged with Golden

    Args:
        df_ (pd.DataFrame): The input DataFrame containing the dataset.
        data_path (str): The path to the data directory.
        golden_csv_path (str): The path to the golden CSV file.
        compare_csv_path (str): The path to the CSV file containing compare images.
        movegoldenimgs (bool): Flag indicating whether to move golden images to images directory.
        savemaplight (str): Flag indicating whether {SaveMapLight} is present in
                    original image paths before pre-processing the dataset paths.

    Returns:
        pandas.DataFrame: A DataFrame containing merged information from the golden and compare files.
    """
    merged_df = pd.DataFrame()
    for proj, tb in df_.groupby(['project', 'TB']).groups.keys():

        goldencsv = getgolden(proj, tb, golden_csv_path, data_path, movegoldenimgs, savemaplight)
        df_proj_tb = df_.loc[(df_['project'] == proj) & (df_['TB'] == tb)]
        comparefile = get_allcomp_csv(df_proj_tb, compare_csv_path, data_path)
        comparefile = aggr_status_across_lights(comparefile)
        comp_on_board_cols = ['CompName', 'directory', 'TOP_BOT']
        compare_light = comparefile.groupby(comp_on_board_cols)[
            'LightingCondition'].apply(' '.join).reset_index()
        comparefile.drop(['LightingCondition'], axis=1, inplace=True)
        if 'Type' in comparefile.columns.tolist():
            comparefile.drop(['Type'], axis=1, inplace=True)
        # comparefile.drop(['Type', 'LightingCondition'], axis=1, inplace=True)
        comparefile = pd.merge(comparefile, compare_light, how='left', on=comp_on_board_cols)
        comparefile = comparefile.drop_duplicates(subset=['CompName', 'BoardSN', 'TOP_BOT', 'directory'])
        comp_const_cols = ['CompName', 'TOP_BOT', 'TB_Panel']
        merged = pd.merge(comparefile, goldencsv[comp_const_cols + ['directory']],
                          how='left', on=comp_const_cols)

        merged['TB_Panel_x'], merged['TB_Panel_y'] = merged['TB_Panel'].copy(), merged['TB_Panel'].copy()
        merged.drop('TB_Panel', axis=1, inplace=True)
        merged['project_name'] = proj
        merged['TB'] = tb

        merged_df = pd.concat([merged_df, merged], axis=0)

    return merged_df


def output_combined_dataset(df_, sampling=False,
                            data_path=None, golden_csv_path=None,
                            compare_csv_path=None, output_dir=None,
                            valid=None, movegoldenimgs=None,
                            project_name=None, savemaplight=None):
    """
    Generate a combined dataset for a specific project and save it as a CSV file.

    Args:
        df_ (pd.DataFrame): The input DataFrame containing the dataset.
        sampling (bool): Flag indicating whether to perform sampling on the dataset.
        data_path (str): The path to the data directory.
        golden_csv_path (str): The path to the golden CSV file.
        compare_csv_path (str): The path to the CSV file containing compare images.
        output_dir (str): The directory where the generated CSV file will be saved.
        valid (bool): Flag indicating whether the dataframe corresponds to train or validation data.
        movegoldenimgs (bool): Flag indicating whether to move golden images to images directory.
        project_name (str): The name of the project.
        savemaplight (str): Flag indicating whether {SaveMapLight} is present in
                    original image paths before pre-processing the dataset paths.

    Returns:
        None
    """
    max_rows = 15000
    if valid:
        df_ = df_[df_['isValid'] == 1]
    else:
        df_ = df_[df_['isValid'] == 0]

    merged_file = getboards_mergedw_golden(df_, golden_csv_path,
                                           compare_csv_path,
                                           data_path,
                                           movegoldenimgs,
                                           savemaplight)

    if sampling and merged_file.shape[0] > max_rows:
        merged_filep = merged_file.loc[
            merged_file['ConfirmDefect'] == 'PASS'].sample(axis=0, n=max_rows)
        merged_filef = merged_file.loc[merged_file['ConfirmDefect'] != 'PASS']
        merged_file = pd.concat([merged_filep, merged_filef], axis=0)

    if merged_file[merged_file['directory_y'].isna()].shape[0]:
        num_na_rows = merged_file[merged_file['directory_y'].isna()].shape[0]
        logging.warning(
            f"\n\nFound {num_na_rows} rows with no golden directory. Removing them"
        )
        prev_rows = merged_file.shape[0]
        merged_file.dropna(how='any', inplace=True)
        logging.warning("Dropped {} Rows due to NA".format(prev_rows - merged_file.shape[0]))

    col = ['directory_x', 'directory_y', 'ConfirmDefect', 'image_name']
    merged_subset = merged_file[col]
    merged_subset['image_name'] = merged_subset.image_name.str.split('_').str[0]
    merged_subset.rename(columns={'directory_x': 'input_path',
                                  'directory_y': 'golden_path', 'image_name': 'object_name',
                                  'ConfirmDefect': 'label'}, inplace=True)
    if valid:
        merged_subset.to_csv(output_dir + '/' + 'valid_combined.csv', index=False)
    else:
        merged_subset.to_csv(output_dir + '/' + 'train_combined.csv', index=False)


def unzip_tars(root_path, projs, top_bot, zip_tar):
    """UnZip Tar files"""
    for proj in projs:
        for tb in top_bot:
            idx = 0
            path = '/'.join([root_path, proj, tb, ''])
            if zip_tar == "zip":
                tars = [x for x in os.listdir(path) if x.endswith('.zip')]
            else:
                tars = [x for x in os.listdir(path) if x.endswith('.tar.gz')]
            for tar in tars:
                fulltar = path + tar
                if os.path.isdir(fulltar[:-7]):
                    continue
                if zip_tar == "zip":
                    shutil.unpack_archive(fulltar, path)
                    # file = tarfile.open(fulltar)
                    # file.extractall(path)
                    # file.close()
                else:
                    import tarfile
                    file = tarfile.open(fulltar)
                    file.extractall(path)
                    file.close()

                idx += 1
            if idx:
                print("Extracted {} tars in {}_{}".format(idx, proj, tb))
            else:
                print("All tars already unzipped ")


def zip_tar_images(df, BOT_TOP, root_data_path):
    """Zip Tar files"""
    if BOT_TOP == 'all':
        # tb_suffs = ['AOI_T', 'AOI_B']
        tb_suffs = ['AOI_T']
        df_combined = df.loc[df['TB'] == 'TOP']
        unzip_tars(root_data_path + 'images', df_combined['project'].unique(), tb_suffs, "zip")

        tb_suffs = ['AOI_B']
        df_combined = df.loc[df['TB'] == 'BOT']
        unzip_tars(root_data_path + 'images', df_combined['project'].unique(), tb_suffs, "zip")
    elif BOT_TOP == 'TOP':
        tb_suffs = ['AOI_T']
        df_combined = df.loc[df['TB'] == 'TOP']
        unzip_tars(root_data_path + 'images', df_combined['project'].unique(), tb_suffs, "tar")
    else:
        tb_suffs = ['AOI_B']
        df_combined = df.loc[df['TB'] == 'BOT']
        unzip_tars(root_data_path + 'images', df_combined['project'].unique(), tb_suffs, "tar")
