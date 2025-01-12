# Paths to the databases used in the experiment
#paths to databases

adi_root=/export/corpora6/ADI17
musan_root=/export/corpora5/JHU/musan
qasr_root=/export/fs06/corpora8/QASR/qasr-speech-corpus-v1.0-release/mgb2.1
mgb3_root=/export/fs05/mkhelfi1/mgb3/mgb3_hf

if [ "$(hostname --domain)" == "clsp.jhu.edu" ];then
  
  adi_root=/export/corpora6/ADI17
  musan_root=/export/corpora5/JHU/musan
  qasr_root=/export/fs06/corpora8/QASR/qasr-speech-corpus-v1.0-release/mgb2.1
  echo 'hello'
  

elif [ "$(hostname --domain)" == "kam.local" ];then

  adi_root=/Users/kamstudy/Documents/Corpora/ADI17
  exit 1

  fi
