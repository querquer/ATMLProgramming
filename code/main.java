package atmlprogramming;

import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.*;

/**
* ATML Programming Task
*
* @author Christian Lausberger, Jannis Becke, Maik Riestock
*
* @version 1.0
*/
public class main {
	
	/**
	* classification
	*
	* @param original data
	*
	* @return filtered data
	*
	*/
	protected static void classify(Instances data) throws Exception {
	    System.out.println("\n1. Classification");

	}
	
	
	/**
	* preprocessing
	*
	* @param original data
	*
	* @return filtered data
	*
	*/
	protected static Instances preprocessing(Instances data) throws Exception {
	    System.out.println("\n1. Preprocessing");
	    
	    
	    // attribute encounter_id, removed 
	    // attribute remove weight, removed
	    // attribute payer_code, removed
	    String[] options_remove = new String[4];
	    options_remove[0] = "-R";                                    // "range"
	    options_remove[1] = "1";                                     // encounter_id
	    options_remove[2] = "6";                                     // weight
	    options_remove[3] = "11";                                    // payer_code
	    
	    Remove remove = new Remove();                         		// new instance of filter
	    remove.setOptions(options_remove);                           // set options
	    remove.setInputFormat(data);                          		// inform filter about dataset **AFTER** setting options
	    
	    Instances data_removed = Filter.useFilter(data, remove);   // apply filter
	    
	    
	    // medical specialty, missing values
	    // Replaces all missing values for nominal and numeric attributes in a dataset with the modes and means from the training data.
	    
	    ReplaceMissingValues replace = new ReplaceMissingValues();
	    replace.setInputFormat(data); 
	    Instances data_removed_and_replaced = Filter.useFilter(data_removed, replace);   // apply filter
	    
	    
	    // This filter removes attributes that do not vary at all or that vary too much. All constant attributes are deleted automatically, 
	    // along with any that exceed the maximum percentage of variance parameter. 
	    // The maximum variance test is only applied to nominal attributes.
	    
	    String[] options_removeUseless = new String[2];
	    options_removeUseless[0] = "-M";                                    	// "range"
	    options_removeUseless[1] = "99.0";                                      // encounter_id
   
	    RemoveUseless removeUseless = new RemoveUseless();                         // new instance of filter
	    remove.setOptions(options_removeUseless);                           		// set options
	    remove.setInputFormat(data);                          						// inform filter about dataset **AFTER** setting options
	    
	    Instances data_removed_and_replaced_rUseless = Filter.useFilter(data_removed_and_replaced, removeUseless);   // apply filter
	    // 26 27 28 3 33 36 37 38 39 40 41 43 44 45 46 47
	    
		return data_removed_and_replaced_rUseless;
	    
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
		Instances data_improved = preprocessing(data);
		
		
	    }	
}
