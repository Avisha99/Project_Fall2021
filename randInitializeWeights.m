function W =  randInitializeWeights(in, out)
W =  zeros(out, 1 +  in);
epsilon_init  =  sqrt(6)/sqrt(in+out);
W =  rand(out, 1 +  in) * 2 * epsilon_init  -epsilon_init;
end