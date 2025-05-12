clc;
clear;
%% Area_level image generation
dz = 'C:\Users\Steven\Downloads\oscillation_detect\DTL-Location-main\DTL-Location-main\DTL FO_Location\Create_Sample\fix_f\sample_data';    % input original data path

% 检查目录是否存在
if ~exist(dz, 'dir')
    error('指定的目录不存在: %s', dz);
end

% 递归获取所有子目录中的 .mat 文件
files = dir(fullfile(dz, '**', '*.mat')); % '**' 表示递归搜索子目录
numberCandidate = numel(files); % 获取文件数量
if numberCandidate == 0
    error('目录中没有找到任何 .mat 文件');
end

imageeeee = [];
tlast = 5;

% 区域划分（假设区域划分为连续的节点编号）
a1 = 1:7;  % 区域 1 的节点编号
a2 = 8:14; % 区域 2 的节点编号
a3 = 15:21; % 区域 3 的节点编号
a4 = 22:29; % 区域 4 的节点编号
aquan = {a1, a2, a3, a4};

parfor j = 1:numberCandidate
    % 获取文件路径
    file_path = fullfile(files(j).folder, files(j).name);
    disp(['正在处理文件: ', file_path]);

    % 加载数据文件
    try
        loaded_data = load(file_path); % 使用 load 加载文件
        if isfield(loaded_data, 'data')
            data = loaded_data.data; % 提取结构体 data
        else
            warning('文件中缺少 data 结构体，跳过此文件: %s', file_path);
            continue;
        end
    catch
        warning('无法加载文件: %s，跳过此文件', file_path);
        continue;
    end

    % 检查 data.p 的大小
    if isfield(data, 'p')
        disp(['data.p 的大小: ', mat2str(size(data.p))]);
    else
        warning('数据文件中缺少 p 变量，跳过此文件: %s', file_path);
        continue;
    end

    % 动态检查并补足缺少的变量
    if ~isfield(data, 'pg')
        % 假设 generator_nodes 是从 1 到 size(data.p, 2)
        generator_nodes = 1:size(data.p, 2);
        data.pg = data.p(:, generator_nodes); % 提取发电机节点的功率数据
    end

    % 提取扰动源位置（从子目录名中获取区域信息）
    [~, subfolder_name] = fileparts(files(j).folder); % 获取子目录名
    source_location = str2double(subfolder_name); % 假设子目录名是数字，表示区域编号
    if isnan(source_location)
        warning('无法解析子目录名中的扰动源位置，跳过此文件: %s', file_path);
        continue;
    end

    % 动态确定区域编号
    region_index = find(cellfun(@(x) ismember(source_location, x), aquan));
    if isempty(region_index)
        warning('扰动源位置 %d 不属于任何区域，跳过此文件: %s', source_location, file_path);
        continue;
    end
    disp(['扰动源位置: ', num2str(source_location)]);
    disp(['所属区域编号: ', num2str(region_index)]);
    disp(['区域节点列表: ', mat2str(aquan{region_index})]);

    % 生成 ttt 的范围
    ttt = 1:min(tlast * 30 + 1, size(data.p, 1));
    disp(['ttt 的范围: ', mat2str([min(ttt), max(ttt)])]);

    % 提取区域数据
    p29 = data.pg(ttt, aquan{region_index});
    if isempty(p29)
        warning('未找到匹配的区域数据，跳过此文件: %s', file_path);
        continue;
    end

    % 生成时频图
    fs = 30;
    D = [];
    for i = 1:size(p29, 2)
        D_i = wvd(p29(:, i), fs, 'smoothedPseudo');
        D = [D, D_i];
    end

    imageeeee = D;
    data_image = imagesc(imageeeee);
    set(gca, 'XTick', [], 'YTick', [], 'XColor', 'none', 'YColor', 'none', 'looseInset', [0 0 0 0]);

    % 构建目标路径
    output_dir = fullfile('e:temp\oscillation_detect\DTL-Location-main\DTL-Location-main\DTL FO_Location\MoreSamples\area_output', num2str(region_index), num2str(source_location));
    if ~exist(output_dir, 'dir')
        mkdir(output_dir); % 如果路径不存在，则创建
    end

    % 保存图像
    output_file = fullfile(output_dir, strcat(files(j).name, '_image', num2str(j), '.png'));
    saveas(data_image, output_file, 'png');
end