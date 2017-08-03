/***********************************************************************

	This file is part of KEEL-software, the Data Mining tool for regression, 
	classification, clustering, pattern mining and so on.

	Copyright (C) 2004-2010

	F. Herrera (herrera@decsai.ugr.es)
    L. Sanchez (luciano@uniovi.es)
    J. Alcala-Fdez (jalcala@decsai.ugr.es)
    S. Garcia (sglopez@ujaen.es)
    A. Fernandez (alberto.fernandez@ujaen.es)
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

package wrapper;

import java.io.IOException;

import org.core.*;

import keel.Algorithms.Decision_Trees.C45.C45;
import java.util.StringTokenizer;
import java.util.ArrayList;
import java.util.Collections;

public class Wrapper {

	myDataset train, val, test;
	String outputTr, outputTst, fileRB, data;
	long startTime, totalTime, seed;
	public static String header;
	int confusionMatrix[][], indexBest, fitness;
	double crossover,mutation;
	boolean ensemble, model, weighting;
	ArrayList <C45> models;
	ArrayList <AUC> aucList;
	private double [] aprioriClassDistribution;
	private double [] weightsAUC;

	//We may declare here the algorithm's parameters
	int populationSize, maxTrials, f1,f2,solSelect,instances;

	private boolean somethingWrong = false; //to check if everything is correct.

	public static final int aucTrain = 0;
	public static final int aucVal = 1;
	public static final int gmVal = 2;
	public static final int ALL = 0;
	public static final int MAJ = 1;


	/**
	 * Default constructor
	 */
	public Wrapper() {
	}

	/**
	 * It reads the data from the input files (training, validation and test) and parse all the parameters
	 * from the parameters array.
	 * @param parameters parseParameters It contains the input files, output files and parameters
	 */
	public Wrapper(parseParameters parameters) {
		this.startTime = System.currentTimeMillis();

		this.train = new myDataset();
		this.val = new myDataset();
		this.test = new myDataset();
		try {
			System.out.println("\nReading the training set: " + parameters.getTrainingInputFile());
			this.train.readClassificationSet(parameters.getTrainingInputFile(), true);
			System.out.println("\nReading the validation set: " + parameters.getValidationInputFile());
			this.val.readClassificationSet(parameters.getValidationInputFile(), false);
			System.out.println("\nReading the test set: " + parameters.getTestInputFile());
			this.test.readClassificationSet(parameters.getTestInputFile(), false);
		}
		catch (IOException e) {
			System.err.println("There was a problem while reading the input data-sets: " + e);
			this.somethingWrong = true;
		}

		//We may check if there are some numerical attributes, because our algorithm may not handle them:
		//somethingWrong = somethingWrong || train.hasNumericalAttributes();
		this.somethingWrong = this.somethingWrong || this.train.hasMissingAttributes();

		this.outputTr = parameters.getTrainingOutputFile();
		this.outputTst = parameters.getTestOutputFile();

		this.fileRB = parameters.getOutputFile(0);
		this.data = parameters.getTrainingInputFile();

		//Now we parse the parameters
		int param = 0;
		seed = Long.parseLong(parameters.getParameter(param++));

		this.populationSize = Integer.parseInt(parameters.getParameter(param++));
		this.maxTrials = Integer.parseInt(parameters.getParameter(param++));
		if (this.populationSize%2 > 0)  this.populationSize++;
		this.crossover = Double.parseDouble(parameters.getParameter(param++)); //crossover probability
		this.mutation = Double.parseDouble(parameters.getParameter(param++));

		String aux2 = parameters.getParameter(param++);

		this.fitness = this.aucVal;
		if (aux2.equalsIgnoreCase("AUC_TR")){
			fitness = this.aucTrain;
		}else if (aux2.equalsIgnoreCase("GM_VAL")){
			fitness = this.gmVal;
		}
		aux2 = parameters.getParameter(param++);
		this.instances = this.ALL;
		if (aux2.equalsIgnoreCase("MAJ")){
			instances = this.MAJ;
		}		
		aux2 = parameters.getParameter(param++);
		ensemble = aux2.equalsIgnoreCase("true");

		aux2 = parameters.getParameter(param++);
		this.weighting = aux2.equalsIgnoreCase("true");
		
		aux2 = parameters.getParameter(param++);
		model = aux2.equalsIgnoreCase("true");

		header = parameters.getTestInputFile();
		String[] aux = null;
		aux = header.split("\\.");
		header = aux[aux.length - 2]; //aux.length-1 is the extension
		aux = header.split("/");
		header = aux[aux.length - 1]; //To be run in SGE    

		Randomize.setSeed(seed);
	}

	/**
	 * It launches the algorithm
	 */
	public void execute() {
		if (this.somethingWrong) { //We do not execute the program
			System.err.println("An error was found, either the data-set has missing values.");
			System.err.println("Please remove the examples with missing data or apply a MV preprocessing.");
			System.err.println("Aborting the program");
			//We should not use the statement: System.exit(-1);
		}
		else {
			//We do here the algorithm's operations

			int nClasses = train.getnClasses();
			aprioriClassDistribution = new double[nClasses];
			for (int i = 0; i < nClasses; i++) {
				aprioriClassDistribution[i] = 1.0 * val.numberInstances(i)/ val.size();
			}

			if (model){ //the model is not previously generated in a file 
				NSGA2 search = new NSGA2(train,seed,populationSize,maxTrials,crossover,mutation,instances,fitness,weighting);
				try{
					search.execute();
				}catch(Exception e){
					e.printStackTrace(System.err);
				}
			}

			//Finally we should fill the training and test output files

			this.generateModel();

			double avgTr = this.doOutput(val, this.outputTr, false);
			double aucTr = getAUC(val);
			double avgTst = this.doOutput(test, this.outputTst, true);
			double aucTst = getAUC(test);
			System.out.print("AUC Train: "+aucTr);
			System.out.println("; AvgAcc Train: "+avgTr);
			System.out.print("AUC Test: "+aucTst);
			System.out.println("; AvgAcc Test: "+avgTst);

			totalTime = System.currentTimeMillis() - startTime;
			System.out.println("Algorithm Finished: "+totalTime);
		}
	}

	private double getAUC(myDataset data){
		Collections.sort(aucList); //order by positive prediction
		int totalClasses = data.getnClasses();
		double [][] aucOk = new double[totalClasses][totalClasses]; 
		double [] tprOk = new double[totalClasses];
		double [][] fprOk = new double[totalClasses][totalClasses];
		double [] tprNew = new double[totalClasses];
		double [][] fprNew = new double[totalClasses][totalClasses];
		double [] tprPrev = new double[totalClasses];
		double [][] fprPrev = new double[totalClasses][totalClasses];
		double pPrev = 0; 
		
		//Test
		for (int i = 0; i < aucList.size(); i++){
			double pAct = aucList.get(i).getPpos();
			if(pAct != pPrev){
				for (int j = 0; j < totalClasses; j++){
					if (data.numberInstances(j) > 0){
						tprNew[j] = tprOk[j]/data.numberInstances(j);
						for (int k = 0; k < totalClasses; k++){
							if ((data.numberInstances(k) > 0)&&(j!=k)){
								fprNew[j][k] = fprOk[j][k]/data.numberInstances(k);
								aucOk[j][k] += (tprPrev[j]+tprNew[j])*(fprNew[j][k]-fprPrev[j][k])/2.0;
								tprPrev[j] = tprNew[j];
								fprPrev[j][k] = fprNew[j][k];
							}
						}
					}	
				}
				pPrev = pAct;
			}
			if(aucList.get(i).getActCl() == aucList.get(i).getPrCl())
				tprOk[aucList.get(i).getPrCl()]++;
			else
				fprOk[aucList.get(i).getPrCl()][aucList.get(i).getActCl()]++;
		}

		for (int j = 0; j < totalClasses; j++){
			if (data.numberInstances(j) > 0){
				tprNew[j] = tprOk[j]/data.numberInstances(j);
				for (int k = 0; k < totalClasses; k++){
					if ((data.numberInstances(k) > 0)&&(j!=k)){
						fprNew[j][k] = fprOk[j][k]/data.numberInstances(k);
						aucOk[j][k] += (tprPrev[j]+tprNew[j])*(fprNew[j][k]-fprPrev[j][k])/2.0;
						aucOk[j][k] += (tprNew[j]+1)*(1-fprNew[j][k])/2.0;
					}
				}
			}	
		}

		double aucFinal = 0;
		int classesSuma = 0;
		for (int j = 0; j < totalClasses; j++){
			if (data.numberInstances(j) > 0){
				for (int k = 0; k < totalClasses; k++){
					if ((data.numberInstances(k) > 0)&&(j!=k)){
						classesSuma++;
						aucFinal += aucOk[j][k];
					}
				}
			}
		}
		
		return (aucFinal/classesSuma);
	}
	
	private double getAUC_OnePoint(myDataset data){
		double auc = 0;
		int totalClasses = 0;
		for (int i = 0; i < this.confusionMatrix.length; i++){
			if (data.numberInstances(i) > 0){
				totalClasses++;
				double tp = 1.0*confusionMatrix[i][i]/data.numberInstances(i);
				for (int j = 0; j < this.confusionMatrix[i].length; j++){
					if ((j != i)&&(data.numberInstances(j) > 0)){
						double fp = 1.0*this.confusionMatrix[j][i]/data.numberInstances(j);
						double auc_j = (tp - fp + 1)/2.0;
						auc += auc_j;
					}
				}
			}
		}		
		double auc2 = (auc/(totalClasses*(totalClasses-1)));
		return auc2;
	}

	private String printConfusionMatrix(){
		String salida = new String("");
		for (int i = 0; i < train.getnClasses(); i++){
			salida += "\n";
			for (int j = 0; j < train.getnClasses(); j++){
				salida += confusionMatrix[i][j]+"\t";
			}
		}
		salida += "\n";
		return salida;
	}

	/**
	 * It selects the best solution according to objective 0 and generates the given RB 
	 * @return the RB with the best value for objective 0
	 */
	private ArrayList <String> getAllSolutions(){
		ArrayList <String> solutions = new ArrayList <String>();
		//This procedure can be updated in order to select any other desirable solution 
		Files function = new Files();
		String funcionStr = function.readFile(header+".var");
		StringTokenizer lines = new StringTokenizer(funcionStr,"\n");

		while(lines.hasMoreTokens()){
			StringTokenizer token = new StringTokenizer(lines.nextToken()," ");
			String solutionFS = token.nextToken();
			solutionFS = solutionFS.replace("\t", "");
			if (solutionFS.contains("1")){
				String solutionIS = token.nextToken();
				solutionIS = solutionIS.replace("\t", "");
				if (solutionIS.contains("1")){
					solutions.add(solutionFS); //111010110101
					solutions.add(solutionIS); //111010110101
				}else{
					System.err.println("Skipping empty solution (FS)");
				}
			}else{
				System.err.println("Skipping empty solution (FS)");
			}

		}
		return solutions;
	}

	private void generateModel(){
		double max_auc = 0;
		ArrayList <String> solutions = this.getAllSolutions();
		models = new ArrayList <C45>(); 

		int nEjemplos = train.getnData();
		if (this.instances == this.MAJ){
			nEjemplos = train.getMajority();
		}
		boolean [] variables = new boolean[train.getnInputs()];
		boolean [] ejemplos = new boolean[nEjemplos];
		this.weightsAUC = new double[solutions.size()/2]; //Hay 2 soluciones FS e IS

		for (int i = 0, j = 0; i < solutions.size(); i+=2, j++){
			int vars, ejs;
			vars = ejs = 0;
			variables = decode(solutions.get(i));
			ejemplos = decode(solutions.get(i+1));
			for (int l = 0; l < variables.length; l++){
				//variables[j] = solution[j];
				if(variables[l]) vars++;
			}
			for (int l = 0; l < ejemplos.length; l++){
				if (ejemplos[l]) ejs++;
			}
			try{
				C45 model = new C45(train,val,test,variables,ejemplos, weighting);

				/***********/
				//double fit = model.getAUCTr();
				model.evaluateTrain();
				double auc_tr = model.getAUCTr();
				if (auc_tr > max_auc){
					max_auc = auc_tr;
					indexBest = j;
				}
				this.weightsAUC[j] = auc_tr;
				model.evaluateTest();
				double auc_tst = model.getAUC();
				System.out.println("Solution["+j+"]:\t"+vars+"\t"+ejs+"\t"+auc_tr+"\t"+auc_tst);

				/***********/				
				models.add(model);
			}catch(Exception e){
				System.err.println("Liada maxima al generar modelo ");
				e.printStackTrace(System.err);
				System.exit(-1);
			}
		}
	}

	private boolean [] decode(String bits){
		boolean [] solution = new boolean[bits.length()]; 
		for (int i = 0; i < solution.length; i++){
			solution[i] = bits.charAt(i) == '1';
		}
		return solution;
	}

	/**
	 * It generates the output file from a given dataset and stores it in a file
	 * @param dataset myDataset input dataset
	 * @param filename String the name of the file
	 */
	private double doOutput(myDataset dataset, String filename, boolean test) {
		String output = new String("");
		confusionMatrix = new int[dataset.getnClasses()][dataset.getnClasses()];
		aucList = new ArrayList<AUC>();
		
		output = dataset.copyHeader(); //we insert the header in the output file
		//We write the output for each example
		for (int i = 0; i < dataset.getnData(); i++) {
			String clReal = dataset.getOutputAsString(i);
			double [] votes = classificationOutput(i,test);
			int index = getOutputTies(votes);
			String clPred = train.getOutputValue(index);
			aucList.add(new AUC(votes[index],dataset.numericClass(clPred),dataset.getOutputAsInteger(i)));
			confusionMatrix[dataset.getOutputAsInteger(i)][dataset.numericClass(clPred)]++;
			output +=  clReal+ " " + clPred + "\n";
		}
		double acc = 0;
		int nClasses = 0;
		for (int i = 0; i < confusionMatrix.length; i++){
			int count = 0; 
			for (int j = 0; j < confusionMatrix[i].length; j++){
				count += confusionMatrix[i][j];
			}
			if (count > 0){
				acc += 1.0*confusionMatrix[i][i]/count;
				nClasses++;
			}
		}
		Files.writeFile(filename, output);
		return acc/nClasses;
		//return 1.0*hits/dataset.size();
	}

	/**
	 * It returns the algorithm classification output given an input example
	 * @param example double[] The input example
	 * @return String the output generated by the algorithm
	 */
	private double [] classificationOutput(int index, boolean test) {
		//String output = new String("?");
		/**
          Here we should include the algorithm directives to generate the
          classification output from the input example
		 */
		double votes[] = new double[train.getnClasses()];

		if (ensemble){

			for (int i = 0; i < models.size(); i++) {
				double [] probs = new double[train.getnClasses()];
				if (test){
					probs = models.get(i).probsTest(index);
				}else{
					probs = models.get(i).probsTrain(index);
				}

				int aggregation = 1; //0 = binary, 1 = weighted, 2 = winner_takes_all
				if(aggregation == 0){
					int maxIndex = 0;
					double max = probs[0];
					for (int j = 1; j < votes.length; j++){
						if (probs[j] > probs[maxIndex]){
							maxIndex = j;
							max = probs[j];
						}
					}
					votes[maxIndex] += 1;
				}else if (aggregation == 1){
					for (int j = 0; j < votes.length; j++){
						votes[j] += probs[j]*this.weightsAUC[j];
					}
				}else{
					int maxIndex = 0;
					double max = probs[0];
					for (int j = 1; j < votes.length; j++){
						if (probs[j] > probs[maxIndex]){
							maxIndex = j;
							max = probs[j];
						}
					}
					votes[maxIndex] += max;
				}
			}
		}else{ //best solution
			//double [] probs = new double[train.getnClasses()];
			if (test){
				votes[models.get(indexBest).classifyTest(index)]++;
			}else{
				votes[models.get(indexBest).classifyTrain(index)]++;
			}
		}
		//return getOutputTies(votes);
		return votes;
	}

	int getOutputTies(double[] max) {
		/*
		 * Tie-breaking step 1: Find out which classes gain the maximum score
		 */
		double maxValue = max[maxIndex(max)];
		double[] ties = new double[max.length];
		for (int i = 0; i < max.length; i++) {
			if (max[i] == maxValue) {
				ties[i] = aprioriClassDistribution[i];
			}
		}

		max = new double[max.length];
		max[maxIndex(ties)] = 1;

		/*
		 * Tie-breaking step 2: Check whether the tying classes have the same a
		 * priori class probability and count these classes.
		 */
		int tieValues = 0;
		maxValue = ties[maxIndex(ties)];
		for (int i = 0; i < ties.length; i++) {
			if (ties[i] == maxValue) {
				tieValues++;
			}
		}

		/*
		 * Tie-breaking step 3: If the tying classes have the same a priori
		 * probabilities, then use randomization to determine the winner among
		 * these classes
		 */
		if (tieValues > 1) {
			tieValues = 0;
			maxValue = ties[maxIndex(ties)];
			int[] stillTying = new int[ties.length];

			for (int i = 0; i < max.length; i++) {
				if (ties[i] == maxValue) {
					stillTying[tieValues] = i;
					tieValues++;
				}
			}
			//return train.getOutputValue(stillTying[Randomize.RandintClosed(0, tieValues-1)]);
			return stillTying[Randomize.RandintClosed(0, tieValues-1)];
		}
		return maxIndex(max);
	}

	static private int maxIndex(int[] array) {
		int max = array[0];
		int index = 0;
		for (int i = 1; i < array.length; i++) {
			if (array[i] > max) {
				max = array[i];
				index = i;
			}
		}
		return index;
	}

	static private int maxIndex(double [] array) {
		double max = array[0];
		int index = 0;
		for (int i = 1; i < array.length; i++) {
			if (array[i] > max) {
				max = array[i];
				index = i;
			}
		}
		return index;
	}
}
