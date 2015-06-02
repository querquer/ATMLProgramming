

import java.util.Iterator;
import java.util.Stack;

import libsvm.svm;
import libsvm.svm_parameter;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.lazy.IB1;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.*;
import weka.filters.unsupervised.instance.RemovePercentage;
import weka.filters.unsupervised.instance.RemoveWithValues;

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
	    
	    // init filter for percentage split
	    RemovePercentage filter = new  RemovePercentage();
	    filter.setInputFormat(data);
	    // split training data (99%)
	    filter.setPercentage(99.0);
	    Instances training = Filter.useFilter(data, filter);
	    // split test data (1%)
	    filter.setPercentage(1.0);
	    Instances test = Filter.useFilter(data, filter);
	    
	    // Naive Bayes
//	    NaiveBayes classifierNaiveBayes = new NaiveBayes(); // init classifier
//	    classifierNaiveBayes.buildClassifier(training); // build classifier
//	    
//	    Evaluation evalNaiveBayes = new Evaluation(test); // init evaluation
//	    evalNaiveBayes.evaluateModel(classifierNaiveBayes, test); // evaluate classifier
//	    System.out.println(evalNaiveBayes.toSummaryString()); // print summary
//	    System.out.println(evalNaiveBayes.toMatrixString()); // print confusion matrix
	    
	    
	    // Ib1
//	    IB1 classifierIb1 = new IB1(); // init classifier
//	    classifierIb1.buildClassifier(training); // build classifier
//	    
//	    Evaluation evalIb1 = new Evaluation(test); // init evaluation
//	    evalIb1.evaluateModel(classifierIb1, test); // evaluate classifier
//	    System.out.println(evalIb1.toSummaryString()); // print summary
//	    System.out.println(evalIb1.toMatrixString()); // print confusion matrix
	    
	    
	    // SVM
	    LibSVM classifierSVM = new LibSVM(); // init classifier
      classifierSVM.buildClassifier(training); // build classifier
      SelectedTag tag2 = new SelectedTag(LibSVM.KERNELTYPE_POLYNOMIAL, LibSVM.TAGS_KERNELTYPE);
      SelectedTag tag = new SelectedTag(LibSVM.SVMTYPE_C_SVC, LibSVM.TAGS_SVMTYPE);
      classifierSVM.setSVMType(tag);
      classifierSVM.setKernelType(tag2);
      classifierSVM.setWeights("1.0 0.5 0.01");
      
	    Evaluation evalSVM = new Evaluation(test); // init evaluation
      evalSVM.evaluateModel(classifierSVM, test); // evaluate classifier
      System.out.println(evalSVM.toSummaryString()); // print summary
      System.out.println(evalSVM.toMatrixString()); // print confusion matrix
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
	    Remove remove = new Remove();                         		// new instance of filter
	    remove.setAttributeIndicesArray(new int[]{0,1,5,10,25,26,27,32,35,36,37,38,39,40,42,43,44,45,46});
	    remove.setInputFormat(data);                          		// inform filter about dataset **AFTER** setting options
	    
	    Instances data_removed = Filter.useFilter(data, remove);   // apply filter
	    
	    // medical specialty, missing values
	    // Replaces all missing values for nominal and numeric attributes in a dataset with the modes and means from the training data.
	    
	    Instances data_removed_and_replaced = null;
	    
      ReplaceMissingValues replace = new ReplaceMissingValues();
      replace.setInputFormat(data_removed); 
      data_removed_and_replaced = Filter.useFilter(data_removed, replace);   // apply filter
	    
		return data_removed_and_replaced;
	    
	}
	
	public static void main(String []args) throws Exception{
		String sourcePath = "data/diabetic_data.csv";
		
		// Reading data
		System.out.println("\n0. Loading data");
		DataSource source = new DataSource(sourcePath);
		Instances data = source.getDataSet();

		if (data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);
		
		// Preprocessing
		Instances data_improved = preprocessing(data);
    System.out.println(data_improved.toSummaryString());
    
		classify(data_improved);
		
	    }	
}
