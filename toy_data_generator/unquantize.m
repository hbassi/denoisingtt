function f = unquantize(f_qtt)

% INPUT: f_qtt --> qtt representation of f
%        npoints --> number of points x
%  
% OUTPUT: x --> grid points
%         f --> fn at grid point

d = f_qtt.d/2;    % depth
n = 2^d-1;
% X = zeros(n,n);
% Y = zeros(n,n);
f = zeros(n,n);

for i = 1:n
    sx=char(num2cell(dec2bin(i,d)));
    sx=reshape(str2num(sx),1,[]);
    for j = 1:n
        %sy = str2double(dec2bin(j,d));
        sy=char(num2cell(dec2bin(j,d)));
        sy=reshape(str2num(sy),1,[]);

        % X(i,j) = bin_to_grid(sx);
        % Y(i,j) = bin_to_grid(sy);

        % sequential: 
        % f(i,j) = f_qtt([sx,sy]+1);

        % interleaved:
        sigma = [sx;sy];
        sigma = sigma(:);
        f(i,j) = f_qtt(sigma+1);
    end
end