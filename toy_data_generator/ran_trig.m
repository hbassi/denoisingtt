function f = ran_trig(X,Y,a,b)

f = zeros(size(X));
N = size(a,1);

for i = 1:N
    for j = 1:N
        f = f + a(i,j)*sin(2*pi*i*(X-0.5)).*sin(2*pi*j*(Y-0.5)) + b(i,j)*cos(2*pi*i*(X-0.5)).*cos(2*pi*j*(Y-0.5));
    end
end