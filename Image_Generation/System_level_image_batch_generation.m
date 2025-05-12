clc;
clear;

%% System_level image generation
dz = 'e:\temp\oscillation_detect\DTL-Location-main\DTL-Location-main\DTL FO_Location\Create_Sample\fix_f\sample_data\'; % input original data path
%files = cellstr(ls([dz '\*.mat']));

files = dir(fullfile(dz, '**', '*.mat'));
files = files(~[files.isdir]); % 排除目录，仅保留文件
 
files2 = cell(size(files));
for i = 1:numel(files)
    abs_path = fullfile(files(i).folder, files(i).name); % 绝对路径
    rel_path = strrep(abs_path, dz, ''); % 转换为相对路径
    rel_path = regexprep(rel_path, ['^' filesep], ''); % 移除开头的文件分隔符
    rel_path = regexprep(rel_path, filesep, '/'); % 统一分隔符为 '/'
    files2{i} = rel_path;
end
files = files2;

numberCandidate = size(files, 1);
imageeeee = [];
tlast = 5;
ttt = 1:1:tlast * 30 + 1;

for j = 1:numberCandidate
    filenmae_i = files{j};
    fullFilePath = fullfile(dz, filenmae_i);

    % 检查文件是否存在
    if ~exist(fullFilePath, 'file')
        warning('文件不存在: %s', fullFilePath);
        continue;
    end

    try
        data = importdata(fullFilePath);
    catch ME
        warning('无法读取文件: %s。错误信息: %s', fullFilePath, ME.message);
        continue;
    end

    label_1 = [24, 11, 19, 23, 12, 25, 26, 5, 14, 6, 27];
    label_2 = [4, 10, 1, 2, 28, 3, 13, 29, 7];
    label_3 = 9;
    label_4 = [18, 8, 16, 17, 15, 20, 21, 22];
    p29 = data.p;
    q29 = data.q;

    % 检查字段是否存在
    if isfield(data, 'w')
        w29 = data.w;
    else
        % 如果字段 "w" 不存在，计算 "w"
        if ~exist('w_warning_shown', 'var') || ~w_warning_shown
            warning('字段 "w" 不存在于文件: %s，正在计算...', fullFilePath);
            w_warning_shown = true; % 设置标志，避免重复警告
        end
        
        % 假设 w 是 p 和 q 的平方和的平方根
        if isfield(data, 'p') && isfield(data, 'q')
            w29 = sqrt(data.p.^2 + data.q.^2);
        else
            error('无法计算字段 "w"，因为字段 "p" 或 "q" 不存在于文件: %s', fullFilePath);
        end
    end

    % 提取与 label_1、label_2、label_3、label_4 对应的列
    p29_1 = p29(:, label_1);
    p29_2 = p29(:, label_2);
    p29_3 = p29(:, label_3);
    p29_4 = p29(:, label_4);

    % 进行 PCA 分析
    [coeff_1, score_1, latent_1] = pca(p29_1);
    [coeff_2, score_2, latent_2] = pca(p29_2);
    [coeff_3, score_3, latent_3] = pca(p29_3);
    [coeff_4, score_4, latent_4] = pca(p29_4);

    pca_size = 1;
    scoreregular_1 = [score_1(:, 1:min(pca_size, size(score_1, 2))), zeros(size(score_1, 1), pca_size - min(pca_size, size(score_1, 2)))];
    scoreregular_2 = [score_2(:, 1:min(pca_size, size(score_2, 2))), zeros(size(score_2, 1), pca_size - min(pca_size, size(score_2, 2)))];
    scoreregular_3 = [score_3(:, 1:min(pca_size, size(score_3, 2))), zeros(size(score_3, 1), pca_size - min(pca_size, size(score_3, 2)))];
    scoreregular_4 = [score_4(:, 1:min(pca_size, size(score_4, 2))), zeros(size(score_4, 1), pca_size - min(pca_size, size(score_4, 2)))];
    fs = 30;

    D_1 = wvd(scoreregular_1, fs, 'smoothedPseudo');
    D_2 = wvd(scoreregular_2, fs, 'smoothedPseudo');
    D_3 = wvd(scoreregular_3, fs, 'smoothedPseudo');
    D_4 = wvd(scoreregular_4, fs, 'smoothedPseudo');

    imageeeee = [D_1, D_2, D_3, D_4];
    data_image = imagesc(imageeeee);
    set(gca, 'XTick', [], 'YTick', [], 'XColor', 'none', 'YColor', 'none', 'looseInset', [0 0 0 0]);

    %source_location_1 = cell2mat(filenmae_i);
    source_location = str2double(filenmae_i(1:2));
    if isempty(source_location)
        source_location = str2double(source_location_1(1));
    end


    label = 0;
    if ~isempty(find(source_location == label_1, 1))
        label = 1;
    elseif ~isempty(find(source_location == label_2, 1))
        label = 2;
    elseif ~isempty(find(source_location == label_3, 1))
        label = 3;
    elseif ~isempty(find(source_location == label_4, 1))
        label = 4;
    end

    % 确保输出目录存在
    outputDir = fullfile('E:\temp\oscillation_detect\DTL-Location-main\DTL-Location-main\DTL FO_Location\MoreSamples\global_output\', num2str(label));
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    datanames = fullfile(outputDir, strcat('image', num2str(j), '.png'));
    saveas(data_image, datanames, 'png');
end