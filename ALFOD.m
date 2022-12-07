%%��Ҫ�ο����£���An Anisotropic Fourth-Order Diffusion Filter for Image Noise Removal 2011��
%%���£���An improved anisotropic diffusion model for detail- and enge-preserving smoothing 2010��
%%���£���An Edge-Adapting Laplacian Kernel For Nonlinear Diffusion Filters 2012��
%%�㷨ԭ���Ľ���1. ��������˹�����滻Ϊ����Ӧ������˹����;�Ľ���2. ��ɢ�������ݶ�Ĥ��ͼ�񷽲ͬȷ��;�Ľ���3. ���뱣���� Lambada*(u-ug);
%%����Ľ׸�������ƫ΢�ַ��̣��ڱ�Ե������������ɢ����ƽ��������ǿ��ɢ�����ƣ���һ����������αӰ�ͱ���ͼ���Ե��
%% ALFODģ��

function main
close all
clear all
clc

addpath(genpath(pwd))

% %% thoracic phantom
% fileName_Nonoise = '.\Thoracic phantom\��Ĥ����.bmp';
% fileName_Noise = '.\Thoracic phantom\�ͼ�����Ĥ����.bmp';

% %% liver image
% fileName_Nonoise = '.\liver image\binzaoNDCT24.png';
% fileName_Noise = '.\liver image\binzaoLDCT24.png';

%% pelvis image
fileName_Nonoise = '.\pelvis image\L006_Full_43.png';
fileName_Noise = '.\pelvis image\L006_Quarter_43.png';

[M, N, c] = size(imread(fileName_Nonoise));
if c > 1
    g= double(rgb2gray(imread(fileName_Nonoise)));   % ������ͼ�� AAPM-Mayo-CT-Challenge�ͼ���CTͼ�� ��ͨ��ͼ��
    I_noi = double(rgb2gray(imread(fileName_Noise)));                     % ����ͼ��
else
    g = double(imread(fileName_Nonoise));    % ��ͨ��ͼ��
    I_noi = double(imread(fileName_Noise));                     % ����ͼ��
end

% g= double(rgb2gray(imread(fileName_Nonoise)));                        % ������ͼ�� AAPM-Mayo-CT-Challenge�ͼ���CTͼ��
% I_noi = double(rgb2gray(imread(fileName_Noise)));                     % ����ͼ��

figure,imshow(I_noi,[],'border','tight');title('�ͼ���CTͼ��');
imcontrast;
figure,imshow(g,[],'border','tight');title('��׼����CTͼ��');
imcontrast;

MSSIM = mssim(g, I_noi);
fprintf('�ṹ���ƶ�Ϊ��%.4f\n',MSSIM);
Psnr = Cal_Psnr(g, I_noi);
fprintf('��ֵ�����Ϊ��%.4f\n',Psnr);
[FSIM, FSIMc ] = FeatureSIM (g, I_noi);
fprintf('�������ƶ�Ϊ��%.4f\n',FSIM);
gmsd = GMSD(g, I_noi);
fprintf('����������ƫ��Ϊ��%.4f\n',gmsd);                  %GMSD �ݶȷ���������ƫ�� ��ֵԽСԽ�ã�

% %��ɢ��ֵ�ļ���
% [Gx,Gy]=gradient(I_noi);
% gradientimage = sqrt(Gx .* Gx + Gy .* Gy);
% gradientimage1=gradientimage(:);
% ind= find(gradientimage1>0);
% kk=gradientimage1(ind);
% kk=kk(:);
% len=size(kk,1);
% kc=sum(kk)/len*0.8;

niter = 20000;   %�Ľ�ƫ΢�ַ��̵ĵ�������������ssim����������ٴε���
del_t = 0.03;    %ʱ�䲽��
gama = 1.5;      % ������˹����ϵ��

% %% Para: thoracic phantom
% %Para of GF
% r = 1; 
% eps = 0.40^2;
% %Para of PDE
% Lambada = 0.01;   % ������Ȩ��ϵ��
% k = 8;

% %% Para: liver image
% %Para of GF
% r = 2;
% eps = 0.07^2;
% %Para of PDE
% Lambada = 0.5;   % ������Ȩ��ϵ��
% k = 11;

%% Para: Pelvis image
%Para of GF
r = 2;
eps = 0.07^2;
%Para of PDE
Lambada = 0.5;   % ������Ȩ��ϵ��
k = 15;

%%
I_LDCT = I_noi / 255;
p = I_LDCT;
I_Fide = guidedfilter(I_LDCT, p, r, eps);
I_Fide = I_Fide * 255;      % �����˲������ͼ��Χ��0-1������Χ��չ��0-255��

tic
%%���������Ľ�ƫ΢��ȥ��
I_out = Ada_ForthOrder(g, I_noi, I_Fide, del_t, niter, k, Lambada, gama);
toc

% ��������ָ��
MSSIM = mssim(g, I_out);
fprintf('�ṹ���ƶ�Ϊ��%.4f\n',MSSIM);
Psnr = Cal_Psnr(g, I_out);
fprintf('��ֵ�����Ϊ��%.4f\n',Psnr);
[FSIM, FSIMc ] = FeatureSIM (g, I_out);
fprintf('�������ƶ�Ϊ��%.4f\n',FSIM);
gmsd = GMSD(g, I_out)   ;
fprintf('����������ƫ��Ϊ��%.4f\n',gmsd);                  %GMSD �ݶȷ���������ƫ�� ��ֵԽСԽ�ã�

figure,imshow(I_out,[],'border','tight');
imcontrast;
end

%% �Ľ׸���������ɢ�㷨
function Denoised_Out = Ada_ForthOrder(I_ori, I_noi, I_Fide, Dt, Item, k, Lambada, gama)
%%ImΪ��������ͼ��DtΪʱ�䲽����ItemΪ����������KcΪ��ɢ��������ɢ�LambadaΪ��ɢ�����еı�����Ĵ�С
niter = Item;
del_t = Dt;
[M,N]=size(I_noi);
SSIM_pro = 0;
Denoised_Out = zeros(N,N);

%����L1��L2������˹����
L1 = [0, 1, 0; 1, -4, 1; 0, 1, 0];
L2 = 0.5 * [1, 0, 1; 0, -4, 0; 1, 0, 1];

L_gL = zeros(M,N);

%%���������Ľ�ƫ΢�ַ���
for iter = 1:niter
    % ��1��������ͼ���ڷ��ߣ���ֱ��Ե�������ߣ����ű�Ե���򣩵Ķ���ƫ����
    ds =1;
    diff_1 =padarray(I_noi,[ds,ds],'symmetric','both');  %ͼ���Ե����
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
    Uxy= (deltaNW + deltaSE - deltaNE -deltaSW)./4;   %% ����һ�׶���ƫ΢������
    
    Ux_2 = Ux.^2;
    Uy_2 = Uy.^2;
    Sum_xy2 = Ux_2 + Uy_2;
    Tidu_Mo = Sum_xy2.^0.5;
    
    Dnn = (Uxx.*Ux_2 + 2.*Ux.*Uy.*Uxy + Uyy.*Uy_2)./(Sum_xy2 + eps);   %% ��ʽ14
    Dtt = (Uxx.*Uy_2 - 2.*Ux.*Uy.*Uxy + Uyy.*Ux_2)./(Sum_xy2 + eps);   %%
    
    maxgradient = max(max(Tidu_Mo));
    variance = stdfilt(I_noi);   %loacal standard deviation
    variance = variance.^2;% loacal variance
    variance=(variance-min(variance(:)))/(max(variance(:))-min(variance(:)))*maxgradient;%

    % ��2�������� c(|deltaU|) ��ɢ������ʽ5
    Cs = k^2./(k^2 + (Tidu_Mo + variance).^2);   %% ��ʽ��5���Ľ��
    
    % ��3�������� L_gL������˹����
    P_Dxx_yy = Cs.^2.*Dnn + Cs.*Dtt;          %%% �Դ���������˹����
    diff_2 = padarray(P_Dxx_yy,[ds,ds],'symmetric','both');  %%�����Ե
    
    %  North, South, East and West pixel
    g_deltaN = diff_2(1:M,  2:N+1);
    g_deltaS = diff_2(3:M+2,2:N+1);
    g_deltaE = diff_2(2:M+1,3:N+2);
    g_deltaW = diff_2(2:M+1,  1:N);
    
    %  �Խ� pixel
    g_deltaNW = diff_2(1:M,   1:N);
    g_deltaSE = diff_2(3:M+2, 3:N+2);
    g_deltaNE = diff_2(1:M,   3:N+2);
    g_deltaSW = diff_2(3:M+2, 1:N);
    
    del_NS = g_deltaN .* g_deltaS;
    del_EW = g_deltaE .* g_deltaW;
    del_U_ij = (max(del_NS, 0) + max(del_EW, 0)) .^0.5;
    C_del = k^2./(k^2 + del_U_ij .^2);   %% ��ʽ��2���͹�ʽ��20���Ľ��
    F_U_ij = 1 - C_del;                  %% ����Ӧ��Ե�����й�ʽ��20��
    for F_i = 1 :M
        for F_j = 1:N
            Lp = L1 - F_U_ij(F_i, F_j) .* ((2 * gama - 1) .* L1 - 2 * gama .* L2);
            L_gL(F_i,F_j)=Lp(1,2) * (g_deltaN(F_i, F_j) + g_deltaS(F_i, F_j) + g_deltaE(F_i, F_j) + g_deltaW(F_i, F_j))+ ...
                Lp(1,1)*(g_deltaNW(F_i, F_j) + g_deltaNE(F_i, F_j) + g_deltaSW(F_i, F_j) + g_deltaSE(F_i, F_j))+...
                Lp(2,2) * P_Dxx_yy(F_i, F_j); %��ʽ(26) ����:An Edge-Adapting Laplacian Kernel For Nonlinear diffusion Filters
        end
    end
    
    % �����㴦����
    I_noi = I_noi -del_t*L_gL - del_t * Lambada*((I_noi - I_Fide));  % �ӱ�����
    
    SSIM = mssim(I_noi, I_ori);
    if(SSIM>SSIM_pro)
        SSIM_pro = SSIM;
        Denoised_Out = I_noi;
    else
        fprintf('��������Ϊ��%d\n',iter);
        break;
    end
end

end