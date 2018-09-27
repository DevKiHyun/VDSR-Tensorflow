%Make Test set5 HR image and Low image(Blur)

target = 'HR';
dataDir = fullfile('Set5\', target);
count = 0;
f_lst = dir(fullfile(dataDir, '*.png'));
folder_label = fullfile('Set5\', 'ground_truth');
folder_input_2x = fullfile('Set5\', 'blur_2x');
folder_input_3x = fullfile('Set5\', 'blur_3x');
folder_input_4x = fullfile('Set5\', 'blur_4x');
mkdir(folder_label);
mkdir(folder_input_2x);
mkdir(folder_input_3x);
mkdir(folder_input_4x);
for f_iter = 1:numel(f_lst)
%     disp(f_iter);
    f_info = f_lst(f_iter);
    if f_info.name == '.'
        continue;
    end
    f_path = fullfile(dataDir,f_info.name);
    disp(f_path);

    img_raw = imread(f_path);
    img_raw = im2double(img_raw);
    
    img_size = size(img_raw);
    width = img_size(2);
    height = img_size(1);
    
    img_raw = img_raw(1:height-mod(height,12),1:width-mod(width,12),:);
    img_size = size(img_raw);
    
    img = img_raw;
    patch_label_name = sprintf('%s/%d',folder_y_ch,count)
    save(sprintf('%s.mat', patch_label_name), 'img' , '-v6');
    
    img = imresize(imresize(img_raw,1/2,'bicubic'),[img_size(1),img_size(2)],'bicubic');
    patch_input_2x_name = sprintf('%s/%d',folder_input_2x,count)
    save(sprintf('%s.mat', patch_input_2x_name), 'img', '-v6');
    
    img = imresize(imresize(img_raw,1/3,'bicubic'),[img_size(1),img_size(2)],'bicubic');
    patch_input_3x_name = sprintf('%s/%d',folder_input_3x,count)
    save(sprintf('%s.mat', patch_input_3x_name), 'img', '-v6');
    
    img = imresize(imresize(img_raw,1/4,'bicubic'),[img_size(1),img_size(2)],'bicubic');
    patch_input_4x_name = sprintf('%s/%d',folder_input_4x,count)
    save(sprintf('%s.mat', patch_input_4x_name), 'img', '-v6');
    
    count = count + 1;
    display(count);
    ``
end