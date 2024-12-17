
d = 8;          % depth in each dimension
n = 2^d;        % # grid points
h = 1./(n-1);   % step-size
xgrid = linspace(0,1,n+1);
xgrid = xgrid(1:n);
[X,Y] = ndgrid(xgrid(1:end-1));

N = 8; % highest frequency

writematrix(X, fullfile('data_rand_trig_f8',sprintf('X_points.csv'))); 
gzip(fullfile('data_rand_trig_f8', 'X_points.csv'));
delete(fullfile('data_rand_trig_f8', 'X_points.csv'));

writematrix(Y, fullfile('data_rand_trig_f8',sprintf('Y_points.csv'))); 
gzip(fullfile('data_rand_trig_f8', 'Y_points.csv'));
delete(fullfile('data_rand_trig_f8', 'Y_points.csv'));

Nsamples = 1;
for p = 1:Nsamples
    tic
    a = rand(N,N); b = rand(N,N); % random amplitudes
    f = ran_trig(X,Y,a,b);
    f = f/max(abs(f),[],'all');
    
    T = quantize(f);
    f_qtt = tt_tensor(T);
    
    tol = 1e-12;
    f8 = round(f_qtt,tol,8);
    f16 = round(f_qtt,tol,16);
    f128 = round(f_qtt,tol,128);
    
    F8 = unquantize(f8);
    F16 = unquantize(f16);
    F128 = unquantize(f128);
    
    writematrix(F8, fullfile('data_rand_trig_f8',sprintf('sample_%i_bd8.csv',p)));
    gzip(fullfile('data_rand_trig_f8',sprintf('sample_%i_bd8.csv',p)));
    delete(fullfile('data_rand_trig_f8',sprintf('sample_%i_bd8.csv',p)));

    writematrix(F16, fullfile('data_rand_trig_f8',sprintf('sample_%i_bd16.csv',p))); 
    gzip(fullfile('data_rand_trig_f8',sprintf('sample_%i_bd16.csv',p)));
    delete(fullfile('data_rand_trig_f8',sprintf('sample_%i_bd16.csv',p)));

    writematrix(F128, fullfile('data_rand_trig_f8',sprintf('sample_%i_bd128.csv',p)));
    gzip(fullfile('data_rand_trig_f8',sprintf('sample_%i_bd128.csv',p)));
    delete(fullfile('data_rand_trig_f8',sprintf('sample_%i_bd128.csv',p)));

    time = toc;
    fprintf('generated sample %i out of %i\n',p,Nsamples)
    fprintf('last sample took %.3f seconds \n',time)
end

%% Plot data
% figure
% contourf(xgrid(1:end-1),xgrid(1:end-1),F8)
% set(gca,'fontsize',16)
% set(gcf,'color','w');
% colorbar
% title('bond dimension 8')
% 
% figure
% contourf(xgrid(1:end-1),xgrid(1:end-1),F16)
% set(gca,'fontsize',16)
% set(gcf,'color','w');
% colorbar
% title('bond dimension 16')
% 
% figure
% contourf(xgrid(1:end-1),xgrid(1:end-1),F128)
% set(gca,'fontsize',16)
% set(gcf,'color','w');
% colorbar
% title('bond dimension 128')
