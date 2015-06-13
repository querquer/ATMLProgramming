
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Random;
import java.util.Scanner;

import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.*;
import weka.filters.unsupervised.instance.*;
import weka.classifiers.Classifier;
import weka.classifiers.CollectiveEvaluation;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.collective.meta.YATSI;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.rules.DecisionTable;


/**
 * ATML Programming Task
 *
 * @author Christian Lausberger, Jannis Becke, Maik Riestock
 *
 * @version 1.0
 */
public class ClassificationWithConsoleInput {

	public enum Classifiers {
		NAIVEBAYES, DECISIONTABLE, SVM, YATSI
	}
	
	private Classifiers selectedClassifier;
	private boolean usePercentageSplit;
	private int trainPercentage, numFolds, unlabeledTrain;
	
	/**
	 * classification
	 *
	 * @param original data
	 * @return filtered data
	 */
	private void classify(Instances data) throws Exception {
		System.out.println("\n2. Classification");
		
		// Validation - Percentage Split
		data.randomize(new Random(1));
		int trainSize = (int) Math.round(data.numInstances() * trainPercentage / 100);
		int testSize = data.numInstances() - trainSize;
		
		Instances train = new Instances(data, 0, trainSize - 1);
		Instances test = new Instances(data, trainSize, testSize);

		// Validation - Cross Validation
		Random seed = new Random(1);
		
		// create classifier
		Classifier classifier;
		
		switch (selectedClassifier) {
		case NAIVEBAYES:
			classifier = getNaiveBayes();
			break;
		case DECISIONTABLE:
			classifier = getDecisionTable();
			break;
		case SVM:
			classifier = getSVM();
			break;
		case YATSI:
			classifier = getYATSI();
			break;
		default:
			System.out.println("Classifier could not be found!");
			return;
		}
		
		Evaluation evaluation = new Evaluation(test);
		
		if(selectedClassifier != Classifiers.YATSI){
			// build classifier
			classifier.buildClassifier(train);
		}else{
			//split train instances in labeled and unlabeled
			int labeledSize = (int) Math.round(train.numInstances() * (100 - unlabeledTrain) / 100);
			int unlabeledSize = train.numInstances() - labeledSize;
			Instances trainUnlabeled = new Instances(train, labeledSize, unlabeledSize);
			
			// build classifier
			((YATSI)classifier).buildClassifier(data, usePercentageSplit ? trainUnlabeled : data);
			evaluation = new CollectiveEvaluation(data);
		}
				
		if (usePercentageSplit) {
			System.out.println("-----------     Percentage Split     -----------");
			evaluation.evaluateModel(classifier, test);
		}else{
			System.out.println("-----------     Cross Validation     -----------");
			evaluation.crossValidateModel(classifier, data, numFolds, seed);
		}
		System.out.println(evaluation.toSummaryString()); // print summary
		System.out.println(evaluation.toMatrixString()); // print confusion matrix
	}
	
	/**
	 * @return selected classifier with configuration
	 */
	private Classifier getNaiveBayes(){
		NaiveBayes classifier = new NaiveBayes();
		classifier.setUseKernelEstimator(true);
		
		System.out.println("###############################   NaiveBayes     ###############################");
		return classifier;
	}
	
	/**
	 * @return selected classifier with configuration
	 */
	private Classifier getDecisionTable(){
		DecisionTable classifier = new DecisionTable();
		
		System.out.println("##########################   Decision Table     ##########################");
		return classifier;
	}
	
	/**
	 * @return selected classifier with configuration
	 * @throws Exception 
	 */
	private Classifier getSVM() throws Exception{
		LibSVM classifier = new LibSVM();
		
		//configure SVM
		SelectedTag kernelType = new SelectedTag(LibSVM.KERNELTYPE_POLYNOMIAL, LibSVM.TAGS_KERNELTYPE);
		SelectedTag svmType = new SelectedTag(LibSVM.SVMTYPE_C_SVC, LibSVM.TAGS_SVMTYPE);
		classifier.setSVMType(svmType);
		classifier.setKernelType(kernelType);
		
		System.out.println("###############################   SVM     ###############################");
		return classifier;
	}
	
	/**
	 * @return selected classifier with configuration
	 */
	private Classifier getYATSI(){
		YATSI classifier = new YATSI();
		
		//configure YATSI
		classifier.setKNN(15);
		
		System.out.println("###############################   YATSI     ###############################");
		return classifier;
	}
	
	/**
	 * preprocessing
	 *
	 * @param original data
	 * @return filtered data
	 */
	private static Instances preprocessing(Instances data) throws Exception {
		System.out.println("\n1. Preprocessing");

		// attribute encounter_id, removed
		// attribute remove weight, removed
		// attribute payer_code, removed
		Remove remove = new Remove(); // new instance of filter
		remove.setAttributeIndicesArray(new int[] { 0, 1, 5, 10, 18, 19, 20, 25, 26, 27, 32, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46 });
		remove.setInputFormat(data); // inform filter about dataset **AFTER** setting options

		Instances data_removed = Filter.useFilter(data, remove); // apply filter

		// medical specialty, missing values
		// Replaces all missing values for nominal and numeric attributes in a
		// dataset with the modes and means from the training data.

		Instances data_removed_and_replaced = null;

		ReplaceMissingValues replace = new ReplaceMissingValues();
		replace.setInputFormat(data_removed);
		data_removed_and_replaced = Filter.useFilter(data_removed, replace); // apply filter

		return data_removed_and_replaced;
	}
	
	/**
	 * constructor for new classification object
	 *
	 * @param selectedClassifier
	 * @param unlabeledTrain TODO
	 * @param percentage split or cross validation
	 * @param amount train and test data
	 * @param number of folds for crossvalidation
	 *            
	 */
	public ClassificationWithConsoleInput(Classifiers selectedClassifier, boolean usePercentageSplit, int trainPercentage, int numFolds, int unlabeledTrain) {
		this.selectedClassifier = selectedClassifier;
		this.usePercentageSplit = usePercentageSplit;
		this.trainPercentage = trainPercentage;
		this.numFolds = numFolds;
		this.unlabeledTrain = unlabeledTrain;
	}

	public static void main(String[] args) throws Exception {
		String dataPath = "data/diabetic_data.csv";
		
		//Console menu
		System.out.println("Select Classifier:");
		System.out.println("1 - Naive Bayes\n2 - Decision Table\n3 - SVM\n4 - YATSI");
		Scanner in = new Scanner(System.in);
		int selectedClassifier = in.nextInt();
		
		System.out.println("Select Evaluation Method:");
		System.out.println("1 - Percentage Split\n2 - Cross Validation");
		int ups = in.nextInt();
		int split = 100, folds = 0;
		if (ups == 1){
			System.out.println("Enter percentage for split (0-100):");
			split = in.nextInt();
		}else{
			System.out.println("Enter number of folds:");
			folds = in.nextInt();
		}

		//yatsi: need percentag for unlabeled train data
		int unlabeledTrain = 0;
		if(Classifiers.values()[selectedClassifier -1] == Classifiers.YATSI){
			System.out.println("Enter percentage for unlabeled training data (0-100):");
			unlabeledTrain = in.nextInt();
		}
		in.close();
		
		ClassificationWithConsoleInput myClassification = new ClassificationWithConsoleInput(Classifiers.values()[selectedClassifier -1], ups == 1 ? true: false, split, folds, unlabeledTrain);
		
		// Reading data
		System.out.println("\n0. Loading data");
		DataSource source = new DataSource(dataPath);
		Instances data = source.getDataSet();
		System.out.println(data.toSummaryString());

		if (data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);

		// Preprocessing
		Instances data_improved = preprocessing(data);
		System.out.println(data_improved.toSummaryString());
		
		myClassification.classify(data_improved);
	}

	/**
	 * evenClasses
	 *
	 * @param original data
	 */
	private static void evenClasses(Instances data) throws Exception {

		RemoveWithValues filter_1 = new RemoveWithValues();

		String[] options_1 = new String[6];
		options_1[0] = "-C";
		options_1[1] = "last";
		options_1[2] = "-S";
		options_1[3] = "0.0";
		options_1[4] = "-L";
		options_1[5] = "2-3";
		filter_1.setOptions(options_1);

		filter_1.setInputFormat(data);
		Instances data_1 = Filter.useFilter(data, filter_1);

		RemoveWithValues filter_2 = new RemoveWithValues();

		String[] options_2 = new String[6];
		options_2[0] = "-C";
		options_2[1] = "last";
		options_2[2] = "-S";
		options_2[3] = "0.0";
		options_2[4] = "-L";
		options_2[5] = "1,3";
		filter_2.setOptions(options_2);

		filter_2.setInputFormat(data);
		Instances data_2 = Filter.useFilter(data, filter_2);

		RemoveWithValues filter_3 = new RemoveWithValues();

		String[] options_3 = new String[6];
		options_3[0] = "-C";
		options_3[1] = "last";
		options_3[2] = "-S";
		options_3[3] = "0.0";
		options_3[4] = "-L";
		options_3[5] = "1-2";
		filter_3.setOptions(options_3);

		filter_3.setInputFormat(data);
		Instances data_3 = Filter.useFilter(data, filter_3);

		float amount_01 = data_1.numInstances();
		float amount_02 = data_2.numInstances();
		float amount_03 = data_3.numInstances();
		float minimumInst = 0;

		if (amount_01 < amount_02 && amount_01 < amount_03) {
			minimumInst = amount_01;
		}
		if (amount_02 < amount_01 && amount_02 < amount_03) {
			minimumInst = amount_02;
		}
		if (amount_03 < amount_01 && amount_03 < amount_02) {
			minimumInst = amount_03;
		}

		RemovePercentage filter_4 = new RemovePercentage();

		String[] options_4 = new String[2];
		options_4[0] = "-P";
		float p_1 = (1 - (minimumInst / amount_01)) * 100;
		options_4[1] = String.valueOf(p_1);

		filter_4.setOptions(options_4);

		filter_4.setInputFormat(data_1);
		Instances data_1_even = Filter.useFilter(data_1, filter_4);

		RemovePercentage filter_5 = new RemovePercentage();

		String[] options_5 = new String[2];
		options_5[0] = "-P";
		float p_2 = (1 - (minimumInst / amount_02)) * 100;
		options_5[1] = String.valueOf(p_2);

		filter_5.setOptions(options_5);

		filter_5.setInputFormat(data_2);
		Instances data_2_even = Filter.useFilter(data_2, filter_5);

		RemovePercentage filter_6 = new RemovePercentage();

		String[] options_6 = new String[2];
		options_6[0] = "-P";
		float p_3 = (1 - (minimumInst / amount_03)) * 100;
		options_6[1] = String.valueOf(p_3);

		filter_6.setOptions(options_6);

		filter_6.setInputFormat(data_3);
		Instances data_3_even = Filter.useFilter(data_3, filter_6);

		String destPath = "data/data_1_even.arff";
		saveAsARFFjava(data_1_even, destPath);
		destPath = "data/data_2_even.arff";
		saveAsARFFjava(data_2_even, destPath);
		destPath = "data/data_3_even.arff";
		saveAsARFFjava(data_3_even, destPath);
	}

	/**
	 * saves a dataset as .arff using java io
	 *
	 * @param dataset
	 * @param destination path
	 *            
	 */
	private static void saveAsARFFjava(Instances data, String destPath) throws Exception {
		System.out.println("\n1.1 Save dataset as ARFF file");
		BufferedWriter writer = new BufferedWriter(new FileWriter(destPath));
		writer.write(data.toString());
		writer.flush();
		writer.close();
	}
}
