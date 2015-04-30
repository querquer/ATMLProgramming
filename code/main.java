package atmlprogramming;

import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;


public class main {
	
	protected static void preprocessing(Instances data) throws Exception {
	    System.out.println("\n1. Preprocessing");

	  }
	
	public static void main(String []args) throws Exception{
		String sourcePath = "D:\\repository\\atmlprogramming\\data\\diabetic_data.csv";
		
		// Reading data
		System.out.println("\n0. Loading data");
		DataSource source = new DataSource(sourcePath);
		Instances data = source.getDataSet();

		if (data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);
		
		// Preprocessing
		preprocessing(data);
		
		
	    }	
}
