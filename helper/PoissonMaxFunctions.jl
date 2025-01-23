








# function probYatIteration(Y,i,dist)
#     num = pdf(dist,Y)*cdf(dist, Y)^(i-1)
#     denom = cdf(dist, Y)^i - cdf(dist, Y-1)^i
#     return num/denom
# end



# function sampleIndexOld(Y,D,dist)
#     #this is an approximation for stability; take out if
#     #testing for correctness
#     #println(pdf(dist,Y))
#     if pdf(dist, Y) < 10e-10 && Y > mean(dist)
#         println(pdf(dist, Y))
#         println("ahhh1")
#         return rand(DiscreteUniform(1,D))
#     end
#     if pdf(dist, Y) < 10e-10 && Y < mean(dist)
#         println("ahhh2")
#         return 1
#     end
#     index = 1
#     b  = []
#     totalmasstried = 0
#     @views for d in D:-1:1
#         b1 = [1-p for p in b]
#         probY = probYatIteration(Y,d,dist)
#         if isempty(b1)
#             stopprobtemp = probY
#         else
#             stopprobtemp = probY * prod(b1)
#         end

#         stopprob = stopprobtemp / (1-totalmasstried)

#         if stopprob > 1 && abs(stopprob - 1) < 10e-5
#             stopprob = 1
#         end
#         try
#             stop = rand(Bernoulli(stopprob))
#             if stop
#                 break
#             end
#             if d == 2
#                 index = D
#                 break
#             end
#             totalmasstried += stopprobtemp 
#             push!(b, probY)
#             index += 1
#         catch ex
#             println(probY)
#             println(stopprobtemp)
#             println(totalmasstried)
#             println(stopprob)
#             println(pdf(dist, Y))
#             println(mean(dist))
#             println("Y: ", Y, " dist: ", dist, " D: ", D)
#             @assert 1 == 2
#         end
        
#     end
#     # end
#     return index
# end