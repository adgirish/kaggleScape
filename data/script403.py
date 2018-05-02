
# coding: utf-8

# I converted the dog breed feature into a dog group feature. This is necessary because there are more than 1000 unique entries in the dog breed feature so the breed feature is difficult to work with. Dog groups (e.g., herding, toy, hound) offer a nice way to reduce the number of categories and it provides insights into the adoption preferences of people. 
# 
# Here are the main findings:
# 
# - The people of Austin like herding and terrier mixes.
# - Pit bulls are misunderstood creatures (low adoption and high euthanasia rates).
# - Pure breeds have below average adoption rates.
# 
# Here is what I did.
# 
# I started with this wiki page that describes the groups of breeds based on the American Kennel Club (AKC). 
# 
# https://en.wikipedia.org/wiki/List_of_dog_breeds_recognized_by_the_American_Kennel_Club
# 
# This wiki list is not complete, so I checked which breed names occur frequently that are not on this list and I looked up their groups and added it to my database. Pit bulls are very frequent but they are not recognized as an official breed by the AKC. So I added ‘Pit Bull’ as its own group, and I also added ‘Mix’ and 'Unknown' groups. I did not look up the group of breeds that fall into the 'Unknown' category. Any contributions and extensions to my database is welcome!
# 
# Then I converted the breed strings into group lists. Here are a few examples:
# 
# German Shepherd/Great Pyrenees => [‘Herding', 'Working']
# 
# Miniature Schnauzer/Miniature Poodle => ['Terrier', 'Non-Sporting']
# 
# Pit Bull Mix => [‘Pit Bull', 'Mix']
# 
# Beagle/German Shepherd => ['Hound', 'Herding']
# 
# Dachshund Mix => [‘Hound', 'Mix']
# 
# Chihuahua Mix => [‘Toy', 'Mix']
# 
# Maltese => [‘Toy']
# 
# Siberian Husky => [‘Working']
# 
# This way I can plot the rates of different outcomes as a function of the groups. 
# 
# 
# Please also check out my other scripts:
# 
# The classifier solution of my team (Kaggle for the paws) is described here:
# 
# https://www.kaggle.com/c/shelter-animal-outcomes/forums/t/22538/solution-of-team-kaggle-for-the-paws-no-outcome-datetime-features
# 
# I performed some data exploration. I studied how the age, gender, and the breed of cats and dogs influences the outcome.
# 
# https://www.kaggle.com/andraszsom/shelter-animal-outcomes/age-gender-and-breed-vs-outcome
# 
# I calculate uncertainty estimates (confidence intervals) for the outcome types. The main question I answer there is this: what is the confidence interval of the true outcome probabilities based on the observed probabilities?   
# 
# https://www.kaggle.com/andraszsom/shelter-animal-outcomes/uncertainty-estimates-of-outcome-types

# In[ ]:


# read the data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import gridspec
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams.update({'font.size': 12})

df = pd.read_csv('../input/train.csv', sep=',')

feature = 'Breed'

feature_values_dog = df.loc[df['AnimalType'] == 'Dog',feature]
outcome_dog = df.loc[df['AnimalType'] == 'Dog','OutcomeType']
outcome_dog = np.array(outcome_dog)

# unique outcomes:
unique_outcomes = np.unique(outcome_dog)


# In[ ]:


# read in a csv file about breeds and the group they belong to
#breeds_group = np.array(pd.read_csv('breed_info/dog_groups.csv', sep=','))

# Unfortunately I don't know how to upload my csv file to kaggle, so I include it below as a string list.

breeds = ['Blue Lacy','Queensland Heeler','Rhod Ridgeback','Retriever','Chinese Sharpei','Black Mouth Cur','Catahoula','Staffordshire','Affenpinscher','Afghan Hound','Airedale Terrier','Akita','Australian Kelpie','Alaskan Malamute','English Bulldog','American Bulldog','American English Coonhound','American Eskimo Dog (Miniature)','American Eskimo Dog (Standard)','American Eskimo Dog (Toy)','American Foxhound','American Hairless Terrier','American Staffordshire Terrier','American Water Spaniel','Anatolian Shepherd Dog','Australian Cattle Dog','Australian Shepherd','Australian Terrier','Basenji','Basset Hound','Beagle','Bearded Collie','Beauceron','Bedlington Terrier','Belgian Malinois','Belgian Sheepdog','Belgian Tervuren','Bergamasco','Berger Picard','Bernese Mountain Dog','Bichon Fris_','Black and Tan Coonhound','Black Russian Terrier','Bloodhound','Bluetick Coonhound','Boerboel','Border Collie','Border Terrier','Borzoi','Boston Terrier','Bouvier des Flandres','Boxer','Boykin Spaniel','Briard','Brittany','Brussels Griffon','Bull Terrier','Bull Terrier (Miniature)','Bulldog','Bullmastiff','Cairn Terrier','Canaan Dog','Cane Corso','Cardigan Welsh Corgi','Cavalier King Charles Spaniel','Cesky Terrier','Chesapeake Bay Retriever','Chihuahua','Chinese Crested Dog','Chinese Shar Pei','Chinook','Chow Chow',"Cirneco dell'Etna",'Clumber Spaniel','Cocker Spaniel','Collie','Coton de Tulear','Curly-Coated Retriever','Dachshund','Dalmatian','Dandie Dinmont Terrier','Doberman Pinsch','Doberman Pinscher','Dogue De Bordeaux','English Cocker Spaniel','English Foxhound','English Setter','English Springer Spaniel','English Toy Spaniel','Entlebucher Mountain Dog','Field Spaniel','Finnish Lapphund','Finnish Spitz','Flat-Coated Retriever','French Bulldog','German Pinscher','German Shepherd','German Shorthaired Pointer','German Wirehaired Pointer','Giant Schnauzer','Glen of Imaal Terrier','Golden Retriever','Gordon Setter','Great Dane','Great Pyrenees','Greater Swiss Mountain Dog','Greyhound','Harrier','Havanese','Ibizan Hound','Icelandic Sheepdog','Irish Red and White Setter','Irish Setter','Irish Terrier','Irish Water Spaniel','Irish Wolfhound','Italian Greyhound','Japanese Chin','Keeshond','Kerry Blue Terrier','Komondor','Kuvasz','Labrador Retriever','Lagotto Romagnolo','Lakeland Terrier','Leonberger','Lhasa Apso','L_wchen','Maltese','Manchester Terrier','Mastiff','Miniature American Shepherd','Miniature Bull Terrier','Miniature Pinscher','Miniature Schnauzer','Neapolitan Mastiff','Newfoundland','Norfolk Terrier','Norwegian Buhund','Norwegian Elkhound','Norwegian Lundehund','Norwich Terrier','Nova Scotia Duck Tolling Retriever','Old English Sheepdog','Otterhound','Papillon','Parson Russell Terrier','Pekingese','Pembroke Welsh Corgi','Petit Basset Griffon Vend_en','Pharaoh Hound','Plott','Pointer','Polish Lowland Sheepdog','Pomeranian','Standard Poodle','Miniature Poodle','Toy Poodle','Portuguese Podengo Pequeno','Portuguese Water Dog','Pug','Puli','Pyrenean Shepherd','Rat Terrier','Redbone Coonhound','Rhodesian Ridgeback','Rottweiler','Russell Terrier','St. Bernard','Saluki','Samoyed','Schipperke','Scottish Deerhound','Scottish Terrier','Sealyham Terrier','Shetland Sheepdog','Shiba Inu','Shih Tzu','Siberian Husky','Silky Terrier','Skye Terrier','Sloughi','Smooth Fox Terrier','Soft-Coated Wheaten Terrier','Spanish Water Dog','Spinone Italiano','Staffordshire Bull Terrier','Standard Schnauzer','Sussex Spaniel','Swedish Vallhund','Tibetan Mastiff','Tibetan Spaniel','Tibetan Terrier','Toy Fox Terrier','Treeing Walker Coonhound','Vizsla','Weimaraner','Welsh Springer Spaniel','Welsh Terrier','West Highland White Terrier','Whippet','Wire Fox Terrier','Wirehaired Pointing Griffon','Wirehaired Vizsla','Xoloitzcuintli','Yorkshire Terrier']
groups = ['Herding','Herding','Hound','Sporting','Non-Sporting','Herding','Herding','Terrier','Toy','Hound','Terrier','Working','Working','Working','Non-Sporting','Non-Sporting','Hound','Non-Sporting','Non-Sporting','Toy','Hound','Terrier','Terrier','Sporting','Working','Herding','Herding','Terrier','Hound','Hound','Hound','Herding','Herding','Terrier','Herding','Herding','Herding','Herding','Herding','Working','Non-Sporting','Hound','Working','Hound','Hound','Working','Herding','Terrier','Hound','Non-Sporting','Herding','Working','Sporting','Herding','Sporting','Toy','Terrier','Terrier','Non-Sporting','Working','Terrier','Working','Working','Herding','Toy','Terrier','Sporting','Toy','Toy','Non-Sporting','Working','Non-Sporting','Hound','Sporting','Sporting','Herding','Non-Sporting','Sporting','Hound','Non-Sporting','Terrier','Working','Working','Working','Sporting','Hound','Sporting','Sporting','Toy','Herding','Sporting','Herding','Non-Sporting','Sporting','Non-Sporting','Working','Herding','Sporting','Sporting','Working','Terrier','Sporting','Sporting','Working','Working','Working','Hound','Hound','Toy','Hound','Herding','Sporting','Sporting','Terrier','Sporting','Hound','Toy','Toy','Non-Sporting','Terrier','Working','Working','Sporting','Sporting','Terrier','Working','Non-Sporting','Non-Sporting','Toy','Terrier','Working','Herding','Terrier','Toy','Terrier','Working','Working','Terrier','Herding','Hound','Non-Sporting','Terrier','Sporting','Herding','Hound','Toy','Terrier','Toy','Herding','Hound','Hound','Hound','Sporting','Herding','Toy','Non-Sporting','Non-Sporting','Toy','Hound','Working','Toy','Herding','Herding','Terrier','Hound','Hound','Working','Terrier','Working','Hound','Working','Non-Sporting','Hound','Terrier','Terrier','Herding','Non-Sporting','Toy','Working','Toy','Terrier','Hound','Terrier','Terrier','Herding','Sporting','Terrier','Working','Sporting','Herding','Working','Non-Sporting','Non-Sporting','Toy','Hound','Sporting','Sporting','Sporting','Terrier','Terrier','Hound','Terrier','Sporting','Sporting','Non-Sporting','Toy']

breeds_group = np.array([breeds,groups]).T
dog_groups = np.unique(breeds_group[:,1])


# In[ ]:


# Convert the breed string into group lists

group_values_dog = []

count = 0

not_found = []

for i in feature_values_dog:
    i = i.replace(' Shorthair','')
    i = i.replace(' Longhair','')
    i = i.replace(' Wirehair','')
    i = i.replace(' Rough','')
    i = i.replace(' Smooth Coat','')
    i = i.replace(' Smooth','')
    i = i.replace(' Black/Tan','')
    i = i.replace('Black/Tan ','')
    i = i.replace(' Flat Coat','')
    i = i.replace('Flat Coat ','')
    i = i.replace(' Coat','')
    
    groups = []
    if '/' in i:
        split_i = i.split('/')
        for j in split_i:
            if j[-3:] == 'Mix':
                breed = j[:-4]               
                if breed in breeds_group[:,0]:
                    indx = np.where(breeds_group[:,0] == breed)[0]
                    groups.append(breeds_group[indx,1][0])
                    groups.append('Mix')
                elif np.any([s.lower() in breed.lower() for s in dog_groups]):
                    find_group = [s if s.lower() in breed.lower() else 'Unknown' for s in dog_groups]                    
                    groups.append(find_group[find_group != 'Unknown'])
                    groups.append('Mix')  
                elif breed == 'Pit Bull':
                    groupd.append('Pit Bull')
                    groups.append('Mix')  
                elif 'Shepherd' in breed:
                    groups.append('Herding')
                    groups.append('Mix')  
                else:
                    not_found.append(breed)
                    groups.append('Unknown')
                    groups.append('Mix')
            else:
                if j in breeds_group[:,0]:
                    indx = np.where(breeds_group[:,0] == j)[0]
                    groups.append(breeds_group[indx,1][0])
                elif np.any([s.lower() in j.lower() for s in dog_groups]):
                    find_group = [s if s.lower() in j.lower() else 'Unknown' for s in dog_groups]                    
                    groups.append(find_group[find_group != 'Unknown'])
                elif j == 'Pit Bull':
                    groups.append('Pit Bull')
                elif 'Shepherd' in j:
                    groups.append('Herding')
                    groups.append('Mix')  
                else:
                    not_found.append(j)
                    groups.append('Unknown')
    else:

        if i[-3:] == 'Mix':
            breed = i[:-4]
            if breed in breeds_group[:,0]:
                indx = np.where(breeds_group[:,0] == breed)[0]
                groups.append(breeds_group[indx,1][0])
                groups.append('Mix')
            elif np.any([s.lower() in breed.lower() for s in dog_groups]):
                find_group = [s if s.lower() in breed.lower() else 'Unknown' for s in dog_groups]                    
                groups.append(find_group[find_group != 'Unknown'])
                groups.append('Mix') 
            elif breed == 'Pit Bull':
                groups.append('Pit Bull')
                groups.append('Mix') 
            elif 'Shepherd' in breed:
                groups.append('Herding')
                groups.append('Mix')  
            else:
                groups.append('Unknown')
                groups.append('Mix') 
                not_found.append(breed)

        else:
            if i in breeds_group[:,0]:
                indx = np.where(breeds_group[:,0] == i)[0]
                groups.append(breeds_group[indx,1][0])
            elif np.any([s.lower() in i.lower() for s in dog_groups]):
                find_group = [s if s.lower() in i.lower() else 'Unknown' for s in dog_groups]                    
                groups.append(find_group[find_group != 'Unknown'])
            elif i == 'Pit Bull':
                groups.append('Pit Bull')
            elif 'Shepherd' in i:
                groups.append('Herding')
                groups.append('Mix') 
            else:
                groups.append('Unknown') 
                not_found.append(i)
    group_values_dog.append(list(set(groups)))

not_f_unique,counts = np.unique(not_found,return_counts=True)

unique_groups, counts = np.unique(group_values_dog,return_counts=True)

# add mix, pit bull, and unknown to the groups
groups = np.unique(np.append(dog_groups,['Mix','Pit Bull','Unknown']))


# In[ ]:


# Calculate rates of different outcomes

outcome_contours = np.zeros([len(unique_outcomes),len(groups),len(groups)])
outcome_counts = np.zeros([len(unique_outcomes),len(groups),len(groups)])

for i in range(len(groups)):
    for j in range(i+1):
        if i == j:
            denominator = group_values_dog.count([groups[i]])
            indices = [k for k, x in enumerate(group_values_dog) if x == [groups[i]]]
            
            sublist = [outcome_dog[ind] for ind in indices]
            
            for k in range(len(unique_outcomes)):
                numerator = sublist.count(unique_outcomes[k])
                outcome_counts[k,i,j] = denominator
                if denominator > 0:
                    outcome_contours[k,i,j] = 1e0 * numerator/denominator
                else:
                    outcome_contours[k,i,j] = 0e0
            
        else:
            denominator = group_values_dog.count([groups[i],groups[j]])
            denominator += group_values_dog.count([groups[j],groups[i]])
            indices = [k for k, x in enumerate(group_values_dog) if x == [groups[i],groups[j]]]
            indices += [k for k, x in enumerate(group_values_dog) if x == [groups[j],groups[i]]]
            
            sublist = [outcome_dog[ind] for ind in indices]
            
            for k in range(len(unique_outcomes)):
                numerator = sublist.count(unique_outcomes[k])
                outcome_counts[k,i,j] = denominator
                outcome_counts[k,j,i] = denominator
                if denominator > 0:
                    outcome_contours[k,i,j] = 1e0 * numerator/denominator
                    outcome_contours[k,j,i] = 1e0 * numerator/denominator
                else:
                    outcome_contours[k,i,j] = 0e0            
                    outcome_contours[k,j,i] = 0e0            


# In[ ]:


# First plot on adoption rates

avg_adoption = np.average(outcome_contours[0,:,:],weights=outcome_counts[0,:,:])                
avg_adoption_group = np.average(outcome_contours[0,:,:],weights=outcome_counts[0,:,:],axis=0)
arg_sort = np.argsort(avg_adoption_group)
avg_count = np.sum(outcome_counts[0,:,:],axis=0)

plt.figure(figsize=(6,6))

plt.subplot(2,1,1)
plt.xlim([-0.5,len(groups)-0.5])
plt.yscale('log')
plt2, = plt.plot(avg_count[arg_sort],'+',mew=2,markersize=10)
plt.ylabel('log10( nr. of dogs )')
plt.xticks(range(len(groups)),'')


plt.subplot(2,1,2)
plt.xlim([-0.5,len(groups)-0.5])
plt.xlabel('Groups')
plt.ylabel('Adoption rate')
plt2, = plt.plot(avg_adoption_group[arg_sort],'+',mew=2,markersize=10)
plt1, = plt.plot(np.arange(-1,len(groups)+1),np.zeros(len(groups)+2)+avg_adoption)
plt.legend([plt1,plt2],['total average adoption rate','adoption rate per group'],loc=4)
plt.xticks(range(len(groups)), groups[arg_sort], rotation='vertical')
plt.tight_layout()
plt.savefig('groups-vs-adoption.jpg',dpi=150)
plt.show()
plt.close()


# The first plot shows the number of dogs (upper panel) and the adoption rate (lower panel) when a group name appears in the group list. That is, some dogs are counted more than once. You see here each group has more than 1000 dogs (the Unknown category is below 1000), which is good. You also see that herding and terrier mixes are popular and pit bulls are not so popular.

# In[ ]:


# Second plot on adoption rates

plt.figure(figsize=(10,6))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.45, 1]) 

plt.subplot(gs[0])
plt.title('Adoption rate')
ax1 = plt.imshow(np.ma.masked_array(outcome_contours[0,:,:],(outcome_contours[0,:,:] >= avg_adoption)),interpolation='nearest',cmap='Blues_r')
cb1 = plt.colorbar(ax1,shrink=.5, pad=.15, aspect=15)
cb1.set_label('Below average adoption rates')
ax2 = plt.imshow(np.ma.masked_array(outcome_contours[0,:,:],(outcome_contours[0,:,:] < avg_adoption)),interpolation='nearest',cmap='Reds')
cb2 = plt.colorbar(ax2,shrink=.5, aspect=15)
cb2.set_label('Above average adoption rates')
plt.xticks(range(len(groups)), groups, rotation='vertical')
plt.yticks(range(len(groups)), groups)

plt.subplot(gs[1])

outcome_counts[0,:,:][outcome_counts[0,:,:] == 0] = 0.1
plt.title('Nr. of dogs')
plt.imshow(np.log10(outcome_counts[0,:,:]),interpolation='nearest',cmap='afmhot',vmin=0e0)
plt.xticks(range(len(groups)), groups, rotation='vertical')
plt.yticks(range(len(groups)), groups)
cb3 = plt.colorbar(shrink=.5, aspect=15)
cb3.set_label('log10( nr. of dogs )')
plt.tight_layout()
plt.savefig('groups-vs-adoption_grid.jpg',dpi=150)
plt.show()
plt.close()


# This second figure shows the different group combinations. That is, each dog is counted only once on this figure. The x and y axes are the groups. The right figure shows the log10( nr of dogs). You can check there for example that there are a large number of dogs in the herding-mix category (light square), and there are a low number of toy-working mixes (dark square). There are no dogs that belong to the mix-mix category. The shelter always assign a likely breed for mixed dogs.
# 
# And the left side shows the adoption rate for the different combinations. The total average adoption rate is around 0.41. Above average adoption rates are red, below average adoption rates are blue. You see that the pit bull row and column are blue (these figures are symmetric along the diagonal). The herding, hound, and working columns and rows are red. What you also see here (which you couldn’t see on the previous plot) is that the diagonal is blue. So pure breeds have a below average adoption rate.

# In[ ]:


# Third plot on all outcomes

fraction_outcomes = np.zeros([11,5])

for i in range(11):
    for j in range(5):
        if i < 10:
            # mix breeds and unknowns
            # off-diagonal elements in a row - remove the diagonal element
            off_diagonal_outcome = np.delete(outcome_contours[j,i,:],i)
            off_diagonal_count = np.delete(outcome_counts[j,i,:],i)
            # weighted sum
            fraction_outcomes[i,j] = np.average(off_diagonal_outcome,weights = off_diagonal_count)
        if i == 10:
            # prue breeds
            # weighted sum
            fraction_outcomes[i,j] = np.average(np.diagonal(outcome_contours[j,:,:]),weights = np.diagonal(outcome_counts[j,:,:]))

groups = np.append(groups,'Pure breeds')

plt.figure(figsize=(7,6))
plt.xlabel('groups')
plt.ylabel('fraction outcomes')
plt.xlim([-0.5,10.5])
plt1 = plt.bar(np.arange(len(groups))-0.25, fraction_outcomes[:,0], 0.5,color='#5A8F29')
plt2 = plt.bar(np.arange(len(groups))-0.25, fraction_outcomes[:,1], 0.5,color='k',bottom = np.sum(fraction_outcomes[:,:1],axis=1))
plt3 = plt.bar(np.arange(len(groups))-0.25, fraction_outcomes[:,2], 0.5,color='#FF8F00',bottom = np.sum(fraction_outcomes[:,:2],axis=1))
plt4 = plt.bar(np.arange(len(groups))-0.25, fraction_outcomes[:,3], 0.5,color='#FFF5EE',bottom = np.sum(fraction_outcomes[:,:3],axis=1))
plt5 = plt.bar(np.arange(len(groups))-0.25, fraction_outcomes[:,4], 0.5,color='#3C7DC4',bottom = np.sum(fraction_outcomes[:,:4],axis=1))
plt.legend([plt1,plt2,plt3,plt4,plt5],unique_outcomes,loc='upper center',fontsize=10,bbox_to_anchor=(0.5, 1.2),
          ncol=3, fancybox=True, shadow=True)
# add mix to the groups where necessary
mix_indcs = [0,1,3,4,5,6,7,9]
for i in range(len(groups)):
    if i in mix_indcs:
        groups[i] = groups[i] + ' Mix'
plt.xticks(range(len(groups)), groups, rotation=90)
plt.tight_layout(pad=4)
plt.savefig('groups-vs-outcome.jpg',dpi=150)
plt.show()


# This plot shows all outcomes for mixed and pure breed groups. It shows that pit bulls' euthanasia rate is at least twice as high as any other group. It also reinforces how low the adoption rate of pit bulls are. 
