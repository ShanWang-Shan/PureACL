#!/bin/bash
declare -A file_dict

file_dict['2017-08-04-V2-Log1-info_files']='Eby22D-rGtZKoXISuPynmg8BtHyu0s5YV0x2wV_6Kef-Rg'
file_dict['2017-08-04-V2-Log3-info_files']='EZbpEOaKIetMjAKUOGCSEjcB-2c_FiPbnuyhSVHvIWaEWg'
file_dict['2017-08-04-V2-Log4-info_files']='EYOqUb_1a8tDsMBoAokV6S8Bln-gtn9fDLaksMuRw0Tm_w'
file_dict['2017-08-04-V2-Log5-info_files']='EYtT85HV3fRFiAFdZWQ7zLMBqqDwvAbTojb2klF63rxp2A'
file_dict['2017-10-26-V2-Log1-info_files']='EWCoEOyzTChMv3HVsmkER5UB3X9IT078rOQz-U7ju8bSCw'
file_dict['2017-10-26-V2-Log3-info_files']='EYFArPf4IOFAgO9ZP-z8tEgBr3FqKz1JhPkdSGWRjlitYA'
file_dict['2017-10-26-V2-Log4-info_files']='EfkruM3-WaVIkvRt7yaMjQcBysrwPIpnh0W9klESmh8DUA'
file_dict['2017-10-26-V2-Log5-info_files']='ESd4FRPRyGtCtbgIUa3wM_kBCzGVSQcseyDdTftfmq0kHg'

file_dict['2017-08-04-V2-Log1-sat18']='ESKlaMf-O5RHmNPi8okrinABcQnSgExZy-5EbW6fhs0XPQ'
file_dict['2017-08-04-V2-Log3-sat18']='EQLBSX0ZWgxAp1XNpLffMHoBQBiEAUiqQXktJegimvhAwA'
file_dict['2017-08-04-V2-Log4-sat18']='EUVR7aEvh9tKgVJjvo4JPskB53mwunNoTZguHTv9JDmWVQ'
file_dict['2017-08-04-V2-Log5-sat18']='EbSMjz-tueRItQszhv5ysQ4B3HrEzHsgWoceKhb5YceXPQ'
file_dict['2017-10-26-V2-Log1-sat18']='EWHc-Sc8LlhAleFK8ZyW9MsBup0TUnELBlg9sim3OJ5LEw'
file_dict['2017-10-26-V2-Log3-sat18']='Eer-nJaT62xPqBGTPsTlUAEBU5R-97rcii2rc3cvN_Sl4w'
file_dict['2017-10-26-V2-Log4-sat18']='EeKCHU9_egVIocQzX-B_FQMBmsDsIVniu-yYaIg2kiXjhA'
file_dict['2017-10-26-V2-Log5-sat18']='Eedfhp8nYh9CqMepag-QgcwBKK_Rao6Q4YGsFhBwplYZ1w'

for key in "${!file_dict[@]}"; do
  link=${file_dict[$key]}
  name=$key'.tar.gz'
  dir=${key: 0: 18}'/'${i: 19}
        if [ ! -d ${i: 0: 18} ]; then
                mkdir ${i: 0: 18}
        fi
        if [ ! -d $dir ]; then
                mkdir $dir
        fi
	echo "Downloading: "$key
        #wget --no-check-certificate 'https://anu365-my.sharepoint.com/:u:/g/personal/u7094434_anu_edu_au/'$link'&download=1'
        wget --no-check-certificate 'https://anu365-my.sharepoint.com/:u:/g/personal/u7094434_anu_edu_au/'$link'?download=1'
        mv $link'?download=1' $name
        tar -zxvf $name -C $dir
        rm $name
done
