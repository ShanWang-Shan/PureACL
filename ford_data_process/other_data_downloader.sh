#!/bin/bash
declare -A file_dict

#file_dict['2017-08-04-V2-Log3-Satellite_Images']='EW9W6xX0sL1FlHX68q7qEj4BORstIRcwYsjpSDyCr8xe2w?e=xuI48N'
#file_dict['2017-08-04-V2-Log4-Satellite_Images']='Eb-gPnnvCcRNmSJdjizckpUBlL9iKd5foDIaWv4O_8RxWg?e=038Oun'
#file_dict['2017-08-04-V2-Log5-Satellite_Images']='Efw-gDdQoBFCkSuuP4B1z8YBGEVKBXOmV4glydIatrO8uw?e=M2aPPT'
#file_dict['2017-08-04-V2-Log6-Satellite_Images']='EWKUCGh4oz5EmNUNGsv-utkBYC4iRN7To27fYulblgv-Hg?e=d7uyI2'
#file_dict['2017-10-26-V2-Log3-Satellite_Images']='EUxO077rflVArAICkjCbF7sB1Ew92TD3VZMFa8w0pntVpg?e=5n1ypI'
#file_dict['2017-10-26-V2-Log4-Satellite_Images']='EYubNdqP8SxIp7TgIWjVlV8B8QWXHJQf_J-_EdVg6RdFGA?e=6Rqflh'
#file_dict['2017-10-26-V2-Log5-Satellite_Images']='EbZwpsD5iGZMisWM8jmMIfkBkVMIR7oveFT-pnGnQINlgA?e=tCt4nO'
#file_dict['2017-10-26-V2-Log6-Satellite_Images']='EWiyWO22b59IvwAyYU6_IlQBvFml_NYwTLUTV99GuM9ULw?e=e0juS5'
#
#file_dict['2017-08-04-V2-Log3-info_files']='EZbpEOaKIetMjAKUOGCSEjcBGQSwizpckJPKlr4s1dH0sA?e=yaoKqg'
#file_dict['2017-08-04-V2-Log4-info_files']='EYOqUb_1a8tDsMBoAokV6S8BzjjwRUrlEPJB_njE5VGKDA?e=Gh2quy'
#file_dict['2017-08-04-V2-Log5-info_files']='EYtT85HV3fRFiAFdZWQ7zLMB5-2sG26MnZEbXiGRpnTxXQ?e=vlJdbg'
#file_dict['2017-08-04-V2-Log6-info_files']='EeDaibG69SxNjNI-DSzQHcMBUOEhiQEGOpqXi0hhyaMltg?e=f41sS3'
#file_dict['2017-10-26-V2-Log3-info_files']='EYFArPf4IOFAgO9ZP-z8tEgBgQdLunDt97nCTLsAYxROUQ?e=EM5y4p'
#file_dict['2017-10-26-V2-Log4-info_files']='EfkruM3-WaVIkvRt7yaMjQcBixEY9mxKUa6EPvllcbcVmQ?e=mu7yP4'
#file_dict['2017-10-26-V2-Log5-info_files']='ESd4FRPRyGtCtbgIUa3wM_kBKNZKfTlU7KPMx1knG1b1jw?e=qciLUz'
#file_dict['2017-10-26-V2-Log6-info_files']='EfIfRoI7S5xBmr3GJvYiBMkBwDKetYgXpJrD1kyLUZphew?e=7PL8B9'
#
#file_dict['2017-08-04-V2-Log3-pcd']='EXJERyuFiGhAiXaxMhFIu9cBmVeFCifTQFkgzxKrXb3zkQ?e=24eHJ9'
#file_dict['2017-08-04-V2-Log4-pcd']='EcDHVG_9YpZNlfL3bLp-180BV-YhPgeUjTWbDgjt4PpCtg?e=fgqotg'
#file_dict['2017-08-04-V2-Log5-pcd']='EVk_8Ah6q_hClNyEaQ7vSmYBYBysVyGZzGJmiSHMte78qg?e=qUalA1'
#file_dict['2017-08-04-V2-Log6-pcd']='EYMPio1gSVFAlGGOKzc9Ff8Bl-KeChtVB8Z6aiyfp1uSMg?e=qX9DxP'
#file_dict['2017-10-26-V2-Log3-pcd']='EW_5iCAmfpBIlatZ7TD7FuoB_8LATyybgjdYf4THWxDNUw?e=l99gsl'
#file_dict['2017-10-26-V2-Log4-pcd']='EXq8jhCk-odBiXlRuIBu71YBFhXbmcSmXmNOITD2yEOEiA?e=W4vXof'
#file_dict['2017-10-26-V2-Log5-pcd']='EWCL-kr6EIhHiEShZpsi2JwB5e4FxyJh01sUDKkYLpsg7Q?e=zBdIVT'
#file_dict['2017-10-26-V2-Log6-pcd']='Ecmi_tkvzWdIsS4hV1A-4xMBvGx1cysWe5HxIueaYnoGuA?e=1g9dlE'

file_dict['2017-08-04-V2-Log4-2dkp']='Edz69U9PM3pLhbxsXlopsxYBhgQ0svcaVhbU-B7gSl_a0w?e=fHFK9o'
file_dict['2017-08-04-V2-Log5-2dkp']='Eeym1zxiu79KtMTUelf6xzYBvFMRgABg5agjtoIB1Xndqw?e=3yZ7eB'
file_dict['2017-10-26-V2-Log4-2dkp']='ET6246soVmxHvtmLlWUd4SkBnRkgMNmmwJOQIPIal2F5FQ?e=DibrD3'
file_dict['2017-10-26-V2-Log5-2dkp']='EWQJcvT3HtxLjH1VsUac0wwBJX6E69ZKQCjVXdZjYxFQSQ?e=Ao5Aux'

file_dict['2017-08-04-V2-Log4-sat18']='EUVR7aEvh9tKgVJjvo4JPskBhxDX-8FmNgdHG0do4-hGrQ?e=XP2cpj'
file_dict['2017-08-04-V2-Log5-sat18']='EbSMjz-tueRItQszhv5ysQ4BHaIjxIDYzWymTdgDkSnedA?e=hlDpbD'
file_dict['2017-10-26-V2-Log4-sat18']='EeKCHU9_egVIocQzX-B_FQMBa3KLcUW5YC7op1t7A45dKA?e=O1khcM'
file_dict['2017-10-26-V2-Log5-sat18']='Eedfhp8nYh9CqMepag-QgcwBgEg9649s6rNwvSFyLxtLqA?e=29I1hu'

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
        wget --no-check-certificate 'https://anu365-my.sharepoint.com/:u:/g/personal/u7094434_anu_edu_au/'$link'&download=1'
        mv $link'&download=1' $name
        tar -zxvf $name -C $dir
        rm $name
done
