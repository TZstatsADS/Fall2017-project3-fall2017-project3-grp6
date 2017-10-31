%%You gonna put folder 'images' in root directory and put this file in root directory to run it. You can also set the working directory by your convenience.
function lbpfeature()
conf.calDir = './' ; % calculating directory
conf.dataDir = './images/' ; % data (image) directory 
conf.outDir = './output/'; % output directory
conf.prefix = 'lbp_' ;
conf.lbpPath = fullfile(conf.outDir, [conf.prefix 'feature.mat']);

imname = dir(strcat(conf.dataDir,'*.jpg'));    
im_num = length(imname);
lbp = zeros(im_num, 59);

for a = 1:length(imname)
    img = imread(fullfile(conf.dataDir,imname(a).name));
    if (size(img,3)>1)% grayscale
        img = rgb2gray(img);
    lbp(a,:) = extractLBPFeatures(img);
    sprintf('%s%d','image',a,'completed')
end

save(conf.lbpPath, 'lbp');
csvwrite('./output/lbp.csv',lbp);
end

