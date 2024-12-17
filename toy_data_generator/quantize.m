function [T] = quantize(f)

n = size(f,1);
d = log2(n+1);

T = zeros(2*ones(1,2*d));

for i = 1:n-1
    sx=char(num2cell(dec2bin(i,d)));
    sx=reshape(str2num(sx),1,[]);
    for j = 1:n-1
        sy=char(num2cell(dec2bin(j,d)));
        sy=reshape(str2num(sy),1,[]);
        
        % sequential: 
        % ind = num2cell([sx(1:end),sy(1:end)]+1);
        % T(ind{:}) = f(i,j);

        % interleaved:
        sigma = [sx;sy];
        sigma = sigma(:);
        ind = num2cell(sigma+1);
        T(ind{:}) = f(i,j);
    end
end

end