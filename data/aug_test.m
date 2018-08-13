target = 'HR';
dataDir = fullfile('Set5\', target);
count = 0;
f_lst = dir(fullfile(dataDir, '*.png'));
folder_y_ch = fullfile('Set5\', 'y_ch');
folder_y_ch_2x = fullfile('Set5\', 'y_ch_2x');
folder_y_ch_3x = fullfile('Set5\', 'y_ch_3x');
folder_y_ch_4x = fullfile('Set5\', 'y_ch_4x');
folder_color = fullfile('Set5\', 'color');
mkdir(folder_y_ch);
mkdir(folder_y_ch_2x);
mkdir(folder_y_ch_3x);
mkdir(folder_y_ch_4x);
mkdir(folder_color);
for f_iter = 1:numel(f_lst)
%     disp(f_iter);
    f_info = f_lst(f_iter);
    if f_info.name == '.'
        continue;
    end
    f_path = fullfile(dataDir,f_info.name);
    disp(f_path);
    img_raw = imread(f_path);
    if size(img_raw,3)==3
        img_raw = rgb2ycbcr(img_raw);
        color = img_raw(:,:, 2:3);
        img_raw = img_raw(:,:,1);
        
%     else
%         img_raw = rgb2ycbcr(repmat(img_raw, [1 1 3]));
    end
    
    img_raw = im2double(img_raw);
    
    img_size = size(img_raw);
    width = img_size(2);
    height = img_size(1);
    
    img_raw = img_raw(1:height-mod(height,12),1:width-mod(width,12),:);
    color = color(1:height-mod(height,12),1:width-mod(width,12),:);
    
    img_size = size(img_raw);
    
    img = img_raw;
    patch_y_ch_name = sprintf('%s/%d',folder_y_ch,count)
    save(sprintf('%s.mat', patch_y_ch_name), 'img' , '-v6');
    
    img = imresize(imresize(img_raw,1/2,'bicubic'),[img_size(1),img_size(2)],'bicubic');
    patch_y_ch_2x_name = sprintf('%s/%d',folder_y_ch_2x,count)
    save(sprintf('%s.mat', patch_y_ch_2x_name), 'img', '-v6');  
    
    img = imresize(imresize(img_raw,1/3,'bicubic'),[img_size(1),img_size(2)],'bicubic');
    patch_y_ch_3x_name = sprintf('%s/%d',folder_y_ch_3x,count)
    save(sprintf('%s.mat', patch_y_ch_3x_name), 'img', '-v6'); 
    
    img = imresize(imresize(img_raw,1/4,'bicubic'),[img_size(1),img_size(2)],'bicubic');
    patch_y_ch_4x_name = sprintf('%s/%d',folder_y_ch_4x,count)
    save(sprintf('%s.mat', patch_y_ch_4x_name), 'img', '-v6');    
    
    img = color;
    patch_color_name = sprintf('%s/%d',folder_color,count)
    save(sprintf('%s.mat', patch_color_name), 'img', '-v6');
    
    count = count + 1;
    display(count);
    
    
end