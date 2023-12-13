%%主要参考文章：“An Anisotropic Fourth-Order Diffusion Filter for Image Noise Removal 2011”
%%文章：“An improved anisotropic diffusion model for detail- and enge-preserving smoothing 2010”
%%文章：“An Edge-Adapting Laplacian Kernel For Nonlinear Diffusion Filters 2012”
%%算法原理：改进点1. 将拉普拉斯算子替换为自适应拉普拉斯算子;改进点2. 扩散函数用梯度膜和图像方差共同确定;改进点3. 加入保真项 Lambada*(u-ug);
%%结合四阶各向异性偏微分方程，在边缘，纹理处减少扩散，在平滑区域增强扩散的优势，进一步抑制条形伪影和保留图像边缘。
%% ALFOD模型

function main
close all
clear all
clc

addpath(genpath(pwd))

% %% thoracic phantom
% fileName_Noise = '.\Thoracic phantom\低剂量体膜数据.bmp';

% %% liver image
% fileName_Noise = '.\liver image\binzaoLDCT24.png';
% test 2
% fileName_Noise = '.\liver image\L004_Quarter_33.png';
% test 3
% fileName_Noise = '.\liver image\L004_Quarter_38.png';

%% pelvis image
% fileName_Noise = '.\pelvis image\L006_Quarter_43.png';
% test 2
% fileName_Noise = '.\pelvis image\L004_Quarter_48.png';
% test 3
fileName_Noise = '.\pelvis image\L006_Quarter_24.png';

[M, N, c] = size(imread(fileName_Noise));
if c > 1
    I_noi = double(rgb2gray(imread(fileName_Noise)));                     % 噪声图像
else
    I_noi = double(imread(fileName_Noise));                     % 噪声图像
end

figure,imshow(I_noi,[],'border','tight');title('低剂量CT图像');
imcontrast;

del_t = 0.03;    %时间步长
gama = 1.5;      % 拉普拉斯算子系数

% %% Para: thoracic phantom
%Para of GF
% r = 1;
% eps = 0.40^2;
% %Para of PDE
% Lambada = 0.01;   % 保真项权重系数
% k = 8;
% niter = 881;

% %% Para: liver image
%Para of GF
% r = 2;
% eps = 0.07^2;
% %Para of PDE
% Lambada = 0.5;   % 保真项权重系数
% k = 11;
% niter = 49;

%% Para: Pelvis image
%Para of GF
r = 2;
eps = 0.07^2;
%Para of PDE
Lambada = 0.5;   % 保真项权重系数
k = 15;
niter = 23;

%%
I_LDCT = I_noi / 255;
p = I_LDCT;
I_Fide = guidedfilter(I_LDCT, p, r, eps);
I_Fide = I_Fide * 255;      % 导向滤波输出的图像范围是0-1，将范围扩展到0-255；

tic
%%各项异性四阶偏微分去噪
I_out = Ada_ForthOrder_NoGroundTruth(I_noi, I_Fide, del_t, niter, k, Lambada, gama);
toc

figure,imshow(I_out,[],'border','tight');
imcontrast;
end

%% 四阶各向异性扩散算法
function Denoised_Out = Ada_ForthOrder_NoGroundTruth(I_noi, I_Fide, Dt, Item, k, Lambada, gama)
%%Im为输入噪声图像，Dt为时间步长，Item为迭代次数，Kc为扩散函数中扩散项；Lambada为扩散函数中的保真项的大小
niter = Item;
del_t = Dt;
[M,N]=size(I_noi);
Denoised_Out = zeros(N,N);

%定义L1和L2拉普拉斯算子
L1 = [0, 1, 0; 1, -4, 1; 0, 1, 0];
L2 = 0.5 * [1, 0, 1; 0, -4, 0; 1, 0, 1];

L_gL = zeros(M,N);

%%各向异性四阶偏微分方程
for iter = 1:niter
    % 第1步，计算图像在法线（垂直边缘）和切线（沿着边缘方向）的二阶偏导数
    ds =1;
    diff_1 =padarray(I_noi,[ds,ds],'symmetric','both');  %图像边缘处理
    % North, South, East and West pixel
    deltaN = diff_1(1:M,  2:N+1);
    deltaS = diff_1(3:M+2,2:N+1);
    deltaE = diff_1(2:M+1,3:N+2);
    deltaW = diff_1(2:M+1,  1:N);
    
    deltaNW = diff_1(1:M,   1:N);
    deltaSE = diff_1(3:M+2, 3:N+2);
    deltaNE = diff_1(1:M,   3:N+2);
    deltaSW = diff_1(3:M+2, 1:N);
    
    Ux = (deltaN-deltaS)./2;
    Uy = (deltaW-deltaE)./2;
    Uxx = deltaN + deltaS - 2.*I_noi;
    Uyy = deltaW + deltaE - 2.*I_noi;
    Uxy= (deltaNW + deltaSE - deltaNE -deltaSW)./4;   %% 计算一阶二阶偏微分算子
    
    Ux_2 = Ux.^2;
    Uy_2 = Uy.^2;
    Sum_xy2 = Ux_2 + Uy_2;
    Tidu_Mo = Sum_xy2.^0.5;
    
    Dnn = (Uxx.*Ux_2 + 2.*Ux.*Uy.*Uxy + Uyy.*Uy_2)./(Sum_xy2 + eps);   %% 公式14
    Dtt = (Uxx.*Uy_2 - 2.*Ux.*Uy.*Uxy + Uyy.*Ux_2)./(Sum_xy2 + eps);   %%
    
    maxgradient = max(max(Tidu_Mo));
    variance = stdfilt(I_noi);   %loacal standard deviation
    variance = variance.^2;% loacal variance
    variance=(variance-min(variance(:)))/(max(variance(:))-min(variance(:)))*maxgradient;%
    
    % 第2步，计算 c(|deltaU|) 扩散函数公式5
    Cs = k^2./(k^2 + (Tidu_Mo + variance).^2);   %% 公式（5）的结合
    
    % 第3步，计算 L_gL拉普拉斯算子
    P_Dxx_yy = Cs.^2.*Dnn + Cs.*Dtt;          %%% 对此求拉普拉斯算子
    diff_2 = padarray(P_Dxx_yy,[ds,ds],'symmetric','both');  %%扩充边缘
    
    %  North, South, East and West pixel
    g_deltaN = diff_2(1:M,  2:N+1);
    g_deltaS = diff_2(3:M+2,2:N+1);
    g_deltaE = diff_2(2:M+1,3:N+2);
    g_deltaW = diff_2(2:M+1,  1:N);
    
    %  对角 pixel
    g_deltaNW = diff_2(1:M,   1:N);
    g_deltaSE = diff_2(3:M+2, 3:N+2);
    g_deltaNE = diff_2(1:M,   3:N+2);
    g_deltaSW = diff_2(3:M+2, 1:N);
    
    del_NS = g_deltaN .* g_deltaS;
    del_EW = g_deltaE .* g_deltaW;
    del_U_ij = (max(del_NS, 0) + max(del_EW, 0)) .^0.5;
    C_del = k^2./(k^2 + del_U_ij .^2);   %% 公式（2）和公式（20）的结合
    F_U_ij = 1 - C_del;                  %% 自适应边缘文章中公式（20）
    for F_i = 1 :M
        for F_j = 1:N
            Lp = L1 - F_U_ij(F_i, F_j) .* ((2 * gama - 1) .* L1 - 2 * gama .* L2);
            L_gL(F_i,F_j)=Lp(1,2) * (g_deltaN(F_i, F_j) + g_deltaS(F_i, F_j) + g_deltaE(F_i, F_j) + g_deltaW(F_i, F_j))+ ...
                Lp(1,1)*(g_deltaNW(F_i, F_j) + g_deltaNE(F_i, F_j) + g_deltaSW(F_i, F_j) + g_deltaSE(F_i, F_j))+...
                Lp(2,2) * P_Dxx_yy(F_i, F_j); %公式(26) 文章:An Edge-Adapting Laplacian Kernel For Nonlinear diffusion Filters
        end
    end
    
    % 最后计算处理结果
    I_noi = I_noi -del_t*L_gL - del_t * Lambada*((I_noi - I_Fide));  % 加保真项
    Denoised_Out = I_noi;
end

end