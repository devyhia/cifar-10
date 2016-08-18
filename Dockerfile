FROM dkarchmervue/python34-opencv3:latest

CMD pip install scipy jupyter matplotlib notebook numpy pandas scikit-learn seaborn six

EXPOSE 8888
