import os

epochs = int(input("How many epochs : "))

subdomain_name = ["habenaria","porsche","rouget","piranha","mulet","requin","atlas","ardennes","raie","perdrix","quiscale","rouloul","sitelle"]
bad_sub = ["barbeau","epipactis","sole"]

subdomain_name = subdomain_name+bad_sub
hostname_known = [subdomain_name[i]+".polytechnique.fr" for i in range(len(subdomain_name))]
hostname_known = hostname_known[:21]


training_set = [
#   [first_dim,second_dim,third_dim,poison_rate]
    [400,200,100,0.0],
]
training_set.sort()
training_set.reverse()


print("Number of training units :",len(training_set))
print("Number of available servers :",len(hostname_known))


for i in range(min(len(hostname_known),len(training_set))):
    #os.system("gnome-terminal -- '/bin/bash -c \"./short/script.sh "+hostname_known[0]+" 1 0 0 200\" '")
    host = hostname_known[i]
    title = ""
    # ________________________________________________________________ Change here __________________________
    os.system(f"gnome-terminal --title=\"{title}\" -- bash -c \"cd ~;./pathtoscript/resserv.sh {host} {epochs}; \"")
    # exec bash -i


for i in range(min(len(hostname_known),len(training_set))):
    #os.system("gnome-terminal -- '/bin/bash -c \"./script.sh "+hostname_known[0]+" 1 0 0 200\" '")
    host = hostname_known[i]
    dim1,dim2,dim3,pois_r = training_set[i]
    title = ""
    title += str(dim1)+":"+str(dim2)+":"+str(dim3)
    if pois_r>0:
        title+=" Flip "+str(int(pois_r*100)/100)
    # ________________________________________________________________ Change here __________________________
    os.system(f"gnome-terminal --title=\"{title}\" -- bash -c \"cd ~;./pathtoscript/script_nn.sh {host} {dim1} {dim2} {dim3} {pois_r} {epochs}; exec bash -i \"")

if len(hostname_known)-len(training_set)<0:
    print(f"Not enough servers available for training, you need {len(training_set)-len(hostname_known)} more servers!")

input("wait")