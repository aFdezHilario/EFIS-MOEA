/***********************************************************************

	This file is part of KEEL-software, the Data Mining tool for regression, 
	classification, clustering, pattern mining and so on.

	Copyright (C) 2004-2010

	F. Herrera (herrera@decsai.ugr.es)
    L. S�nchez (luciano@uniovi.es)
    J. Alcal�-Fdez (jalcala@decsai.ugr.es)
    S. Garc�a (sglopez@ujaen.es)
    A. Fern�ndez (alberto.fernandez@ujaen.es)
    J. Luengo (julianlm@decsai.ugr.es)

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see http://www.gnu.org/licenses/

 **********************************************************************/

package keel.Algorithms.Decision_Trees.C45;

import java.io.*;
import wrapper.myDataset;
import java.math.*;


/**
 * Class to implement the C4.5 algorithm
 * 
 * @author Cristobal Romero Morales (UCO) (30-03-06)
 * @author modified by Alberto Fernandez (UGR)
 * @version 1.2 (29-04-10)
 * @since JDK 1.5
 *</p>
 **/
public class C45 extends Algorithm {
	/** Decision tree. */
	private Tree root;


	/** Is the tree pruned or not. */
	private boolean prune = true;

	/** Confidence level. */
	private float confidence = 0.25f;

	/** Minimum number of itemsets per leaf. */
	private int minItemsets = 2;

	/** The prior probabilities of the classes. */
	private double[] priorsProbabilities;

	/** Resolution of the margin histogram. */
	private static int marginResolution = 500;

	/** Cumulative margin classification. */
	private double marginCounts[];

	/** The sum of counts for priors. */
	private double classPriorsSum;

	private int tp,tn,fp,fn;
	private int [][] confusionMatrixTr, confusionMatrixTst, confusionMatrixModel;

	public C45(myDataset train, boolean variables[], boolean ejemplos[]) throws Exception {
		try {
			modelDataset = new Dataset(train, variables, ejemplos);
			trainDataset = new Dataset(train, variables);
			testDataset = new Dataset(train, variables);
			priorsProbabilities = new double[modelDataset.numClasses()];
			priorsProbabilities();
			marginCounts = new double[marginResolution + 1];

			// generate the tree
			generateTree(modelDataset);
			evaluateTest();
			evaluateModel();
		} catch (Exception e) {
			e.printStackTrace(System.err);;
			System.exit( -1);
		}
	}

	public C45(myDataset train, myDataset val, myDataset test, boolean variables[], boolean ejemplos[]) throws Exception {
		try {
			modelDataset = new Dataset(train, variables, ejemplos);
			trainDataset = new Dataset(val, variables);
			testDataset = new Dataset(test, variables);

			priorsProbabilities = new double[modelDataset.numClasses()];
			priorsProbabilities();
			marginCounts = new double[marginResolution + 1];
			// generate the tree
			generateTree(modelDataset);
			evaluateTest();
			evaluateTrain();
			evaluateModel();
			evaluateTrain();

		} catch (Exception e) {
			System.err.println("Something is wrong (Generate Final Model) "+e.toString());
			System.exit( -1);
		}
	}

	public void setOptions(StreamTokenizer tokens){
		//empty
	}

	public void generateTree() throws Exception{
		try{
			generateTree(modelDataset);
		}catch (Exception e) {
			System.err.println(e.getMessage());
			System.exit( -1);
		}
	}

	/** Generates the tree.
	 *
	 * @param itemsets		The dataset used to build the tree.
	 *
	 * @throws Exception	If the tree cannot be built.
	 */
	public void generateTree(Dataset itemsets) throws Exception {
		SelectCut selectCut;

		selectCut = new SelectCut(minItemsets, itemsets);
		root = new Tree(selectCut, prune, confidence);
		root.buildTree(itemsets);
	}

	/** Function to evaluate the class which the itemset must have according to the classification of the tree.
	 *
	 * @param itemset		The itemset to evaluate.
	 *
	 * @return				The index of the class index predicted.
	 */
	public double evaluateItemset(Itemset itemset) throws Exception {
		Itemset classMissing = (Itemset) itemset.copy();
		double prediction = 0;
		classMissing.setDataset(itemset.getDataset());
		classMissing.setClassMissing();

		double[] classification = classificationForItemset(classMissing);
		prediction = maxIndex(classification);
		updateStats(classification, itemset, itemset.numClasses());

		//itemset.setPredictedValue( prediction );

		return prediction;
	}

	/** Updates all the statistics for the current itemset.
	 *
	 * @param predictedClassification	Distribution of class values predicted for the itemset.
	 * @param itemset					The itemset.
	 * @param nClasses					The number of classes.
	 *
	 */
	private void updateStats(double[] predictedClassification, Itemset itemset,
			int nClasses) {
		int actualClass = (int) itemset.getClassValue();

		if (!itemset.classIsMissing()) {
			updateMargins(predictedClassification, actualClass, nClasses);

			// Determine the predicted class (doesn't detect multiple classifications)
			int predictedClass = -1;
			double bestProb = 0.0;

			for (int i = 0; i < nClasses; i++) {
				if (predictedClassification[i] > bestProb) {
					predictedClass = i;
					bestProb = predictedClassification[i];
				}
			}

			// Update counts when no class was predicted
			if (predictedClass < 0) {
				return;
			}

			double predictedProb = Math.max(Double.MIN_VALUE,
					predictedClassification[actualClass]);
			double priorProb = Math.max(Double.MIN_VALUE,
					priorsProbabilities[actualClass] /
					classPriorsSum);
		}
	}

	/** Returns class probabilities for an itemset.
	 *
	 * @param itemset		The itemset.
	 *
	 * @throws Exception	If cannot compute the classification.
	 */
	public final double[] classificationForItemset(Itemset itemset) throws
	Exception {
		return root.classificationForItemset(itemset);
	}

	/** Update the cumulative record of classification margins.
	 *
	 * @param predictedClassification	Distribution of class values predicted for the itemset.
	 * @param actualClass				The class value.
	 * @param nClasses					Number of classes.
	 */
	private void updateMargins(double[] predictedClassification,
			int actualClass, int nClasses) {
		double probActual = predictedClassification[actualClass];
		double probNext = 0;

		for (int i = 0; i < nClasses; i++) {
			if ((i != actualClass) && ( //Comparators.isGreater( predictedClassification[i], probNext ) ) )
					predictedClassification[i] > probNext)) {
				probNext = predictedClassification[i];
			}
		}

		double margin = probActual - probNext;
		int bin = (int) ((margin + 1.0) / 2.0 * marginResolution);
		marginCounts[bin]++;
	}

	/** Evaluates if a string is a boolean value.
	 *
	 * @param value		The string to evaluate.
	 *
	 * @return			True if value is a boolean value. False otherwise.
	 */
	private boolean isBoolean(String value) {
		if (value.equalsIgnoreCase("TRUE") || value.equalsIgnoreCase("FALSE")) {
			return true;
		} else {
			return false;
		}
	}

	/** Returns index of maximum element in a given array of doubles. First maximum is returned.
	 *
	 * @param doubles		The array of elements.
	 *
	 */
	public static int maxIndex(double[] doubles) {
		double maximum = 0;
		int maxIndex = 0;

		for (int i = 0; i < doubles.length; i++) {
			if ((i == 0) || //
					doubles[i] > maximum) {
				maxIndex = i;
				maximum = doubles[i];
			}
		}

		return maxIndex;
	}

	/** Sets the class prior probabilities.
	 *
	 * @throws Exception	If cannot compute the probabilities.
	 */
	public void priorsProbabilities() throws Exception {
		for (int i = 0; i < modelDataset.numClasses(); i++) {
			priorsProbabilities[i] = 1;
		}

		classPriorsSum = modelDataset.numClasses();

		for (int i = 0; i < modelDataset.numItemsets(); i++) {
			if (!modelDataset.itemset(i).classIsMissing()) {
				try {
					priorsProbabilities[(int) modelDataset.itemset(i).
					                    getClassValue()] += modelDataset.itemset(i).
					                    getWeight();
					classPriorsSum += modelDataset.itemset(i).getWeight();
				} catch (Exception e) {
					System.err.println(e.getMessage());
				}
			}
		}
	}

	/** Writes the tree and the results of the training and the test in the file.
	 *
	 * @exception 	If the file cannot be written.
	 */
	public void printResult() throws IOException {
		long totalTime = (System.currentTimeMillis() - startTime) / 1000;
		long seconds = totalTime % 60;
		long minutes = ((totalTime - seconds) % 3600) / 60;
		String tree = "";
		PrintWriter resultPrint;

		tree += toString();
		tree += "\n@TotalNumberOfNodes " + root.NumberOfNodes;
		tree += "\n@NumberOfLeafs " + root.NumberOfLeafs;
		tree += "\n@TotalNumberOfNodes " + root.NumberOfNodes;
		int atts = root.getAttributesPerRule();
		if (atts > 0){
			tree += "\n@NumberOfAntecedentsByRule "+(1.0*atts)/root.NumberOfLeafs;
		}else{
			tree += "\n@NumberOfAntecedentsByRule 0";
		}

		tree += "\n\n@NumberOfItemsetsTraining " + trainDataset.numItemsets();
		tree += "\n@NumberOfCorrectlyClassifiedTraining " + correct;
		tree += "\n@PercentageOfCorrectlyClassifiedTraining " +
				(float) (correct * 100.0) / (float) trainDataset.numItemsets() +
				"%";
		tree += "\n@NumberOfInCorrectlyClassifiedTraining " +
				(trainDataset.numItemsets() - correct);
		tree += "\n@PercentageOfInCorrectlyClassifiedTraining " +
				(float) ((trainDataset.numItemsets() - correct) * 100.0) /
				(float) trainDataset.numItemsets() + "%";

		tree += "\n\n@NumberOfItemsetsTest " + testDataset.numItemsets();
		tree += "\n@NumberOfCorrectlyClassifiedTest " + testCorrect;
		tree += "\n@PercentageOfCorrectlyClassifiedTest " +
				(float) (testCorrect * 100.0) / (float) testDataset.numItemsets() +
				"%";
		tree += "\n@NumberOfInCorrectlyClassifiedTest " +
				(testDataset.numItemsets() - testCorrect);
		tree += "\n@PercentageOfInCorrectlyClassifiedTest " +
				(float) ((testDataset.numItemsets() - testCorrect) * 100.0) /
				(float) testDataset.numItemsets() + "%";
		//tree += "\n\n@Variables " +variables.length+" / "+root.numAtts();

		tree += "\n\n@ElapsedTime " +
				(totalTime - minutes * 60 - seconds) / 3600 + ":" +
				minutes / 60 + ":" + seconds;


		resultPrint = new PrintWriter(new FileWriter(resultFileName));
		resultPrint.print(getHeader() + "\n@decisiontree\n\n" + tree);
		resultPrint.close();

	}

	public void setTreeOutput(String resultFileName){
		this.resultFileName = resultFileName;
	}

	public int classifyTrain(int index){
		int cl = 0;
		try {
			Itemset itemset = trainDataset.itemset(index);
			cl = (int) evaluateItemset(itemset);
		}catch (Exception e) {
			System.err.println(e.getMessage());
		}
		return cl;
	}

	public int classifyTest(int index){
		int cl = 0;
		try {
			Itemset itemset = testDataset.itemset(index);
			cl = (int) evaluateItemset(itemset);
		}catch (Exception e) {
			System.err.println(e.getMessage());
		}
		return cl;
	}

	public double [] probsTrain(int index){
		double [] probs = new double[trainDataset.numClasses()];
		try {
			Itemset itemset = trainDataset.itemset(index);
			probs = this.classificationForItemset(itemset);
		}catch (Exception e) {
			System.err.println(e.getMessage());
		}
		return probs;
	}

	public double [] probsTest(int index){
		double [] probs = new double[trainDataset.numClasses()];
		try {
			Itemset itemset = testDataset.itemset(index);
			probs = this.classificationForItemset(itemset);
		}catch (Exception e) {
			System.err.println(e.getMessage());
		}
		return probs;
	}

	/** Evaluates the training dataset and writes the results in the file.
	 *
	 * @exception 	If the file cannot be written.
	 */
	public void printTrain() {
		String text = getHeader();
		confusionMatrixTr = new int[trainDataset.itemset(0).numClasses()][trainDataset.itemset(0).numClasses()];

		for (int i = 0; i < trainDataset.numItemsets(); i++) {
			try {
				Itemset itemset = trainDataset.itemset(i);
				int cl = (int) evaluateItemset(itemset);

				confusionMatrixTr[(int)itemset.getValue(trainDataset.getClassIndex())][cl]++;
				if (cl == (int) itemset.getValue(trainDataset.getClassIndex())) {
					correct++;
				}

				text += trainDataset.getClassAttribute().value(((int) itemset.
						getClassValue())) + " " + trainDataset.getClassAttribute().value(cl)
						+ "\n";
			} catch (Exception e) {
				System.err.println(e.getMessage());
			}
		}


		try {
			PrintWriter print = new PrintWriter(new FileWriter(
					trainOutputFileName));
			print.print(text);
			print.close();
		} catch (IOException e) {
			System.err.println("Can not open the training output file: " +
					e.getMessage());
		}

	}

	public void setTrainOutput(String trainOutputFileName){
		this.trainOutputFileName = trainOutputFileName;
	}

	/** Evaluates the test dataset and writes the results in the file.
	 *
	 * @exception 	If the file cannot be written.
	 */
	public void printTest() {
		String text = getHeader();
		confusionMatrixTst = new int[testDataset.itemset(0).numClasses()][testDataset.itemset(0).numClasses()];

		for (int i = 0; i < testDataset.numItemsets(); i++) {
			try {
				int cl = (int) evaluateItemset(testDataset.itemset(i));
				Itemset itemset = testDataset.itemset(i);
				confusionMatrixTst[(int)itemset.getValue(testDataset.getClassIndex())][cl]++;
				if (cl == (int) itemset.getValue(testDataset.getClassIndex())) {
					testCorrect++;
				}else{
				}

				text += testDataset.getClassAttribute().value(((int) itemset.
						getClassValue())) + " " + testDataset.getClassAttribute().value(cl)
						+ "\n";
			} catch (Exception e) {
				System.err.println(e.getMessage());
			}
		}
		try {
			PrintWriter print = new PrintWriter(new FileWriter(
					testOutputFileName));
			print.print(text);
			print.close();
		} catch (IOException e) {
			System.err.println("Can not open the training output file.");
		}

	}

	public void setTestOutput(String testOutputFileName){
		this.testOutputFileName = testOutputFileName;
	}

	/** Evaluates the test dataset and writes the results in the file.
	 *
	 * @exception 	If the file cannot be written.
	 */
	public void evaluateTest() {

		confusionMatrixTst = new int[testDataset.itemset(0).numClasses()][testDataset.itemset(0).numClasses()];

		for (int i = 0; i < testDataset.numItemsets(); i++) {
			try {
				int cl = (int) evaluateItemset(testDataset.itemset(i));
				Itemset itemset = testDataset.itemset(i);

				confusionMatrixTst[(int)itemset.getValue(testDataset.getClassIndex())][cl]++;
			} catch (Exception e) {
				System.err.println(e.getMessage());
			}
		}
		
	}
	
	/** Evaluates the test dataset and writes the results in the file.
	 *
	 * @exception 	If the file cannot be written.
	 */
	public void evaluateModel() {
		confusionMatrixModel = new int[modelDataset.itemset(0).numClasses()][modelDataset.itemset(0).numClasses()];
		for (int i = 0; i < modelDataset.numItemsets(); i++) {
			try {
				int cl = (int) evaluateItemset(modelDataset.itemset(i));
				Itemset itemset = modelDataset.itemset(i);
				confusionMatrixModel[(int)itemset.getValue(modelDataset.getClassIndex())][cl]++;

			} catch (Exception e) {
				System.err.println(e.getMessage());
			}
		}
	}
	
	/** Evaluates the test dataset and writes the results in the file.
	 *
	 * @exception 	If the file cannot be written.
	 */
	public void evaluateTrain() {
		confusionMatrixTr = new int[trainDataset.itemset(0).numClasses()][trainDataset.itemset(0).numClasses()];
		for (int i = 0; i < trainDataset.numItemsets(); i++) {
			try {
				int cl = (int) evaluateItemset(trainDataset.itemset(i));
				Itemset itemset = trainDataset.itemset(i);
				confusionMatrixTr[(int)itemset.getValue(trainDataset.getClassIndex())][cl]++;

			} catch (Exception e) {
				System.err.println(e.getMessage());
			}
		}
	}


	public double getAUC(){
		double auc = 0;
		int totalClasses = 0;
		for (int i = 0; i < this.confusionMatrixTst.length; i++){
			if (this.testDataset.size(i) > 0){
				totalClasses++;
				double tp = 1.0*confusionMatrixTst[i][i]/this.testDataset.size(i);
				for (int j = 0; j < this.confusionMatrixTst[i].length; j++){
					if ((j != i)&&(this.testDataset.size(j) > 0)){
						double fp = 1.0*this.confusionMatrixTst[j][i]/this.testDataset.size(j);
						double auc_j = (tp - fp + 1)/2.0;
						auc += auc_j;
					}
				}
			}
		}		
		double auc2 = (auc/(totalClasses*(totalClasses-1)));
		return auc2;
	}
	
	public double getAUCModel(){
		double auc = 0;
		int totalClasses = 0;
		for (int i = 0; i < this.confusionMatrixModel.length; i++){
			if (this.modelDataset.size(i) > 0){
				totalClasses++;
				double tp = 1.0*confusionMatrixModel[i][i]/this.modelDataset.size(i);
				for (int j = 0; j < this.confusionMatrixModel[i].length; j++){
					if ((j != i)&&(this.modelDataset.size(j) > 0)){
						double fp = 1.0*this.confusionMatrixModel[j][i]/this.modelDataset.size(j);
						double auc_j = (tp - fp + 1)/2.0;
						auc += auc_j;
					}
				}
			}
		}
		double auc2 = (auc/(totalClasses*(totalClasses-1)));
		return auc2;
	}
	
	public double getAUCTr(){
		double auc = 0;
		int totalClasses = 0;
		for (int i = 0; i < this.confusionMatrixTr.length; i++){
			if (this.trainDataset.size(i) > 0){
				totalClasses++;
				double tp = 1.0*confusionMatrixTr[i][i]/this.trainDataset.size(i);
				for (int j = 0; j < this.confusionMatrixTr[i].length; j++){
					if ((j != i)&&(this.trainDataset.size(j) > 0)){
						double fp = 1.0*this.confusionMatrixTr[j][i]/this.trainDataset.size(j);
						double auc_j = (tp - fp + 1)/2.0;
						auc += auc_j;
					}
				}
			}
		}
		double auc2 = (auc/(totalClasses*(totalClasses-1)));
		return auc2;
	}


	public double getAvgAcc(){
		double avgacc = 0;
		int totalClasses = 0;
		for (int i = 0; i < this.confusionMatrixTst.length; i++){
			if (this.testDataset.size(i) > 0){
				totalClasses++;
				avgacc += 1.0*confusionMatrixTst[i][i]/this.testDataset.size(i);
			}
		}
		return (avgacc/totalClasses);
	}

	public double getGM(){
		double tpr = 1.*tp/(tp+fn);
		double tnr = 1.*tn/(tn+fp);
		return Math.sqrt(tpr*tnr);
	}


	/** Function to print the tree.
	 *
	 */
	public String toString() {
		return root.toString();
	}
}

