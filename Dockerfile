# step1: ��������ʹ��tensorflow-gpu����Ȼ����Ҳ����ʹ��python��Ϊ�������񣬺����ٰ�װtensorflow-gpu������
FROM hub.data.wust.edu.cn:30880/library/tensorflow:1.14.0-gpu-py3

# step2: ����������Ļ���ѧϰ��ص��ļ���������mnist�ļ��У����Ƶ�����ĳ��Ŀ¼�У����磺/home/mnist
COPY ./ /home/dureader

# step3 ���������еĹ���Ŀ¼��ֱ���л���/home/mnistĿ¼��
WORKDIR /home/dureader

# step4 ��װ����
#RUN pip install -r requirements.txt

# step5 ������������ʱ�����������������ֱ������python����
ENTRYPOINT ["sh", "/home/dureader/tensorflow/run.sh"]
