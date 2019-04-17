clear
load('USPS.mat');

p = [10; 50; 100; 200];
[m, n] = size(p);

figure
title("A1")
hold on
A1 = reshape(A(1,:), 16, 16);
imshow(A1')
hold off

figure
title("A2")
hold on
A2 = reshape(A(2,:), 16, 16);
imshow(A2')
hold off

for i=1:m
    [residuals,reconstructed] = pcares(A,p(i));
    
    figure
    hold on
    title("A1 Reconstructed with p = " + p(i))
    A1_reconstruct = reshape(reconstructed(1,:), 16, 16);
    imshow(A1_reconstruct')
    norm(A1_reconstruct-A1)
    hold off

    figure
    hold on
    title("A2 Reconstructed with p = " + p(i))
    A2_reconstruct = reshape(reconstructed(2,:), 16, 16);
    norm(A2_reconstruct-A2)
    imshow(A2_reconstruct')
    hold off
end