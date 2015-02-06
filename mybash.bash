res=1

for (( k=5; k<=10; k+=5 ))
do
        for (( k2=10; k2<=10; k2+=5 ))
        do
                for eta in 0.001,0.01,0.1
                do
                        for lambda in 0.01 0.1 1.0
                        do
                                ./train reviews_Street\,_Surf_\&_Skate.votes.gz duplicate_list.txt.gz duplicate_image_list.txt.gz image_features_Street_Surf_Skate.b also_viewed.txt.gz $k $k2 $eta $lambda 1000 1> imagemodel.res${res}.txt 2>&1
                                let res=${res}+1
                        done
                done
        done
done
