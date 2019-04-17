x = [0, -1, -3, 1, 3];
y = [0, 2, 6, -2, -6];

[n, m] = size(x);


scatter(x,y, 'filled')
for i=1:m
    text(x(i),y(i),"   (" + x(i) + "," + y(i) + ")")
end
% text(-1,2,"   (-1,2)")