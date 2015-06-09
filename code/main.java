
import java.io.File;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Random;

import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.*;
import weka.filters.unsupervised.instance.*;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.ZeroR;
import libsvm.svm;

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
	 * @param original
	 *            data
	 *
	 * @return filtered data
	 *
	 */
	protected static void classify(Instances data) throws Exception {
		System.out.println("\n2. Classification");

		// Validation - Percentage Split
		RemovePercentage filter = new RemovePercentage();
		filter.setInputFormat(data);

		filter.setPercentage(80.0);
		Instances train = Filter.useFilter(data, filter);

		filter.setInvertSelection(true);
		Instances test = Filter.useFilter(data, filter);

		// Validation - Cross Validation
		int folds = 10;
		Random seed = new Random(1);

		// NaiveBayes
		System.out.println("#######################################   NaiveBayes     #######################################");
		NaiveBayes classifierNaiveBayes = new NaiveBayes(); // init classifier
		classifierNaiveBayes.buildClassifier(train); // build classifier

		System.out.println("-----------     Percentage Split     -----------");
		Evaluation evalNaiveBayes = new Evaluation(test); // init evaluation
		evalNaiveBayes.evaluateModel(classifierNaiveBayes, test); // evaluate
																	// classifier
		System.out.println(evalNaiveBayes.toSummaryString()); // print summary
		System.out.println(evalNaiveBayes.toMatrixString()); // print confusion
																// matrix

		System.out.println("-----------     Cross Validation     -----------");
		Evaluation evalCrossNaiveBayes = new Evaluation(data); // init
																// evaluation
		evalCrossNaiveBayes.crossValidateModel(classifierNaiveBayes, data, folds, seed); // evaluate
																							// classifier
		System.out.println(evalCrossNaiveBayes.toSummaryString()); // print
																	// summary
		System.out.println(evalCrossNaiveBayes.toMatrixString()); // print
																	// confusion
																	// matrix

		// Ib1
		System.out.println("#######################################   Ib1     #######################################");
		IBk classifierIbk = new IBk(); // init classifier
		classifierIbk.buildClassifier(train); // build classifier

		System.out.println("-----------     Percentage Split     -----------");
		// Evaluation evalIbk = new Evaluation(test); // init evaluation
		// evalIbk.evaluateModel(classifierIbk, test); // evaluate classifier
		// System.out.println(evalIbk.toSummaryString()); // print summary
		// System.out.println(evalIbk.toMatrixString()); // print confusion
		// matrix

		// System.out.println("-----------     Cross Validation     -----------");
		// Evaluation evalCrossIbk = new Evaluation(data); // init evaluation
		// evalCrossIbk.crossValidateModel(classifierIbk, data, folds, seed); //
		// evaluate classifier
		// System.out.println(evalCrossIbk.toSummaryString()); // print summary
		// System.out.println(evalCrossIbk.toMatrixString()); // print confusion
		// matrix

		// ZeroR
		System.out.println("#######################################   ZeroR     #######################################");
		ZeroR classifierZeroR = new ZeroR(); // init classifier
		classifierZeroR.buildClassifier(train); // build classifier

		System.out.println("-----------     Percentage Split     -----------");
		Evaluation evalZeroR = new Evaluation(test); // init evaluation
		evalZeroR.evaluateModel(classifierZeroR, test); // evaluate classifier
		System.out.println(evalZeroR.toSummaryString()); // print summary
		System.out.println(evalZeroR.toMatrixString()); // print confusion
														// matrix

		System.out.println("-----------     Cross Validation     -----------");
		Evaluation evalCrossZeroR = new Evaluation(data); // init evaluation
		evalCrossZeroR.crossValidateModel(classifierZeroR, data, folds, seed); // evaluate
																				// classifier
		System.out.println(evalCrossZeroR.toSummaryString()); // print summary
		System.out.println(evalCrossZeroR.toMatrixString()); // print confusion
																// matrix

		// DecisionTable
		System.out.println("#######################################   DecisionTable     #######################################");
		DecisionTable classifierDecisionTable = new DecisionTable(); // init
																		// classifier
		classifierDecisionTable.buildClassifier(train); // build classifier

		System.out.println("-----------     Percentage Split     -----------");
		Evaluation evalDecisionTable = new Evaluation(test); // init evaluation
		evalDecisionTable.evaluateModel(classifierDecisionTable, test); // evaluate
																		// classifier
		System.out.println(evalDecisionTable.toSummaryString()); // print
																	// summary
		System.out.println(evalDecisionTable.toMatrixString()); // print
																// confusion
																// matrix

		// System.out.println("-----------     Cross Validation     -----------");
		// Evaluation evalCrossDecisionTable= new Evaluation(data); // init evaluation
		// evalCrossDecisionTable.crossValidateModel(classifierDecisionTable,
		// data, folds, seed); // evaluate classifier
		// System.out.println(evalCrossDecisionTable.toSummaryString()); //print summary
		// System.out.println(evalCrossDecisionTable.toMatrixString()); // print confusion matrix

		
		// LibSVM
		System.out.println("#######################################   LibSVM     #######################################");
		LibSVM classifierSVM = new LibSVM(); // init classifier
		classifierSVM.buildClassifier(train); // build classifier
		SelectedTag tag2 = new SelectedTag(LibSVM.KERNELTYPE_POLYNOMIAL, LibSVM.TAGS_KERNELTYPE);
		SelectedTag tag = new SelectedTag(LibSVM.SVMTYPE_C_SVC, LibSVM.TAGS_SVMTYPE);
		classifierSVM.setSVMType(tag);
		classifierSVM.setKernelType(tag2);
		classifierSVM.setWeights("1.0 0.5 0.01");

		System.out.println("-----------     Percentage Split     -----------");
		Evaluation evalSVM = new Evaluation(test); // init evaluation
		evalSVM.evaluateModel(classifierSVM, test); // evaluate classifier
		System.out.println(evalSVM.toSummaryString()); // print summary
		System.out.println(evalSVM.toMatrixString()); // print confusion matrix

		System.out.println("-----------     Cross Validation     -----------");
		Evaluation evalCrossSVM = new Evaluation(data); // init evaluation
		evalCrossSVM.crossValidateModel(classifierSVM, data, folds, seed); // evaluate classifier
		System.out.println(evalCrossSVM.toSummaryString()); // print summary
		System.out.println(evalCrossSVM.toMatrixString()); // print confusion
															// matrix
		
		// SMO
//		System.out.println("#######################################   SMO     #######################################");
//		SMO classifierSMO = new SMO();
//		classifierSMO.buildClassifier(train);
//
//		System.out.println("-----------     Percentage Split     -----------");
//		Evaluation evalSMO = new Evaluation(test); // init evaluation
//		evalSMO.evaluateModel(classifierSMO, test); // evaluate classifier
//		System.out.println(evalSMO.toSummaryString()); // print summary
//		System.out.println(evalSMO.toMatrixString()); // print confusion matrix
//
//		System.out.println("-----------     Cross Validation     -----------");
//		Evaluation evalCrossSMO = new Evaluation(data); // init evaluation
//		evalCrossSMO.crossValidateModel(classifierSMO, data, folds, seed); // evaluate classifier
//		System.out.println(evalCrossSMO.toSummaryString()); // print summary
//		System.out.println(evalCrossSMO.toMatrixString()); // print confusion matrix

		// YATSI
//		System.out.println("#######################################   YATSI     #######################################");
//
//		// Validation - Percentage Split
//		RemovePercentage unlabeled_filter = new RemovePercentage();
//		unlabeled_filter.setInputFormat(train);
//
//		unlabeled_filter.setPercentage(50.0);
//		Instances train_labeled = Filter.useFilter(train, unlabeled_filter);
//
//		unlabeled_filter.setInvertSelection(true);
//		Instances train_unlabeled = Filter.useFilter(train, unlabeled_filter);
//
//		YATSI classifierYATSI = new YATSI(); // init classifier
//		classifierYATSI.setKNN(15);
//		classifierYATSI.setNoWeights(false);
//		classifierYATSI.buildClassifier(train_labeled, train_unlabeled); // build
//																			// classifier
//
//		System.out.println("-----------     Percentage Split     -----------");
//		CollectiveEvaluation evalYATSI = new CollectiveEvaluation(data); // init
//																			// evaluation
//		evalYATSI.evaluateModel(classifierYATSI, train); // evaluate classifier
//		System.out.println(evalYATSI.toSummaryString()); // print summary
//		System.out.println(evalYATSI.toMatrixString()); // print confusion
//														// matrix
//
//		System.out.println("-----------     Cross Validation     -----------");
//		CollectiveEvaluation evalCrossYATSI = new CollectiveEvaluation(data); // init
//																				// evaluation
//		evalCrossYATSI.crossValidateModel(classifierYATSI, train, folds, seed); // evaluate
//																				// classifier
//		System.out.println(evalCrossYATSI.toSummaryString()); // print summary
//		System.out.println(evalCrossYATSI.toMatrixString()); // print confusion matrix

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
		Remove remove = new Remove(); // new instance of filter
		remove.setAttributeIndicesArray(new int[] { 0, 1, 5, 10, 18, 19, 20, 25, 26, 27, 32, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46 });
		remove.setInputFormat(data); // inform filter about dataset **AFTER**
										// setting options

		Instances data_removed = Filter.useFilter(data, remove); // apply filter

		// medical specialty, missing values
		// Replaces all missing values for nominal and numeric attributes in a
		// dataset with the modes and means from the training data.

		Instances data_removed_and_replaced = null;

		ReplaceMissingValues replace = new ReplaceMissingValues();
		replace.setInputFormat(data_removed);
		data_removed_and_replaced = Filter.useFilter(data_removed, replace); // apply
																				// filter

		return data_removed_and_replaced;

	}

	/**
	 * evenClasses
	 *
	 * @param original
	 *            data
	 */
	protected static void evenClasses(Instances data) throws Exception {

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
	 * saves a dataset as .arff using weka ArffSaver
	 *
	 * @param dataset
	 * @param destination
	 *            path
	 */
	protected static void saveAsARFFweka(Instances data, String destPath) throws Exception {
		System.out.println("\n1.1 Save dataset as ARFF file");

		ArffSaver saver = new ArffSaver();
		saver.setInstances(data);
		saver.setFile(new File(destPath));
		saver.setDestination(new File(destPath));
		saver.writeBatch();
	}

	/**
	 * saves a dataset as .arff using java io
	 *
	 * @param dataset
	 * @param destination
	 *            path
	 */
	protected static void saveAsARFFjava(Instances data, String destPath) throws Exception {
		System.out.println("\n1.1 Save dataset as ARFF file");
		BufferedWriter writer = new BufferedWriter(new FileWriter(destPath));
		writer.write(data.toString());
		writer.flush();
		writer.close();
	}

	public static void main(String[] args) throws Exception {
		String sourcePath_orig = "data/diabetic_data_two_classes.csv";
		// String sourcePath_orig = "data/diabetic_data.csv";
		String sourcePath_improved = "data/data_improved.arff";
		String sourcePath_even = "data/data_even.arff";
		String destPath = "data/data_improved.arff";

		// Reading data
		System.out.println("\n0. Loading data");
		DataSource source = new DataSource(sourcePath_orig);
		Instances data = source.getDataSet();
		System.out.println(data.toSummaryString());

		if (data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);

		// Preprocessing
		Instances data_improved = preprocessing(data);
		System.out.println(data_improved.toSummaryString());

		// saveAsARFFweka(data_improved, destPath);
		// saveAsARFFjava(data_improved, destPath);

		// evenClasses(data_improved);

		classify(data_improved);
	}

}
