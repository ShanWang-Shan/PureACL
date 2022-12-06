#!/bin/bash

files=(2017-10-26/V2/Log3/2017-10-26-V2-Log3-FL.tar.gz
2017-10-26/V2/Log3/2017-10-26-V2-Log3-RR.tar.gz
2017-10-26/V2/Log3/2017-10-26-V2-Log3-SL.tar.gz
2017-10-26/V2/Log3/2017-10-26-V2-Log3-SR.tar.gz
2017-10-26/V2/Log4/2017-10-26-V2-Log4-FL.tar.gz
2017-10-26/V2/Log4/2017-10-26-V2-Log4-RR.tar.gz
2017-10-26/V2/Log4/2017-10-26-V2-Log4-SL.tar.gz
2017-10-26/V2/Log4/2017-10-26-V2-Log4-SR.tar.gz
2017-10-26/V2/Log5/2017-10-26-V2-Log5-FL.tar.gz
2017-10-26/V2/Log5/2017-10-26-V2-Log5-RR.tar.gz
2017-10-26/V2/Log5/2017-10-26-V2-Log5-SL.tar.gz
2017-10-26/V2/Log5/2017-10-26-V2-Log5-SR.tar.gz
2017-10-26/V2/Log6/2017-10-26-V2-Log6-FL.tar.gz
2017-10-26/V2/Log6/2017-10-26-V2-Log6-RR.tar.gz
2017-10-26/V2/Log6/2017-10-26-V2-Log6-SL.tar.gz
2017-10-26/V2/Log6/2017-10-26-V2-Log6-SR.tar.gz
2017-08-04/V2/Log3/2017-08-04-V2-Log3-FL.tar.gz
2017-08-04/V2/Log3/2017-08-04-V2-Log3-RR.tar.gz
2017-08-04/V2/Log3/2017-08-04-V2-Log3-SL.tar.gz
2017-08-04/V2/Log3/2017-08-04-V2-Log3-SR.tar.gz
2017-08-04/V2/Log4/2017-08-04-V2-Log4-FL.tar.gz
2017-08-04/V2/Log4/2017-08-04-V2-Log4-RR.tar.gz
2017-08-04/V2/Log4/2017-08-04-V2-Log4-SL.tar.gz
2017-08-04/V2/Log4/2017-08-04-V2-Log4-SR.tar.gz
2017-08-04/V2/Log5/2017-08-04-V2-Log5-FL.tar.gz
2017-08-04/V2/Log5/2017-08-04-V2-Log5-RR.tar.gz
2017-08-04/V2/Log5/2017-08-04-V2-Log5-SL.tar.gz
2017-08-04/V2/Log5/2017-08-04-V2-Log5-SR.tar.gz
2017-08-04/V2/Log6/2017-08-04-V2-Log6-FL.tar.gz
2017-08-04/V2/Log6/2017-08-04-V2-Log6-RR.tar.gz
2017-08-04/V2/Log6/2017-08-04-V2-Log6-SL.tar.gz
2017-08-04/V2/Log6/2017-08-04-V2-Log6-SR.tar.gz)


wget 'https://ford-multi-av-seasonal.s3-us-west-2.amazonaws.com/Calibration/Calibration-V2.tar.gz'
tar -zxvf 'Calibration-V2.tar.gz' -C './'
rm 'Calibration-V2.tar.gz'
for i in ${files[@]}; do
        shortname=${i: 19}
        fullname=$i
        dir=${i: 19: 18}'/'${i: 19: 21}
        if [ ! -d ${i: 19: 18} ]; then
                mkdir ${i: 19: 18}
        fi
        if [ ! -d $dir ]; then
                mkdir $dir
        fi
	echo "Downloading: "$shortname
        wget 'https://ford-multi-av-seasonal.s3-us-west-2.amazonaws.com/'$fullname
        tar -zxvf $shortname -C $dir
        rm $shortname
done
