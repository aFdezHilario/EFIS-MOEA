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

/**
 * <p>Title: Dataset</p>
 *
 * <p>Description: It contains the methods to read a Classification/Regression Dataset</p>
 *
 *
 * <p>Company: KEEL </p>
 *
 * @author Alberto Fernandez
 * @version 1.0
 */

import java.io.IOException;
import java.lang.String;
import java.util.ArrayList;
import java.util.Vector;

import keel.Dataset.*;

public class myDataset {

	public static final int REAL = 0;
	public static final int INTEGER = 1;
	public static final int NOMINAL = 2;

	private double[][] X = null; //examples array
	private boolean[][] missing = null; //possible missing values
	private boolean[] nominal = null; //nominal attributes
	private boolean[] integer = null; //integer attributes
	private int[] outputInteger = null; //output of the data-set as integer values
	private double[] outputReal = null; //output of the data-set as double values
	private String[] output = null; //output of the data-set as string values
	private double[] emax; //max value of an attribute
	private double[] emin; //min value of an attribute

	private int nData; // Number of examples
	private int nVars; // Number of variables
	private int nInputs; // Number of inputs
	private int nClasses; // Number of outputs

	public InstanceSet IS; //The whole instance set

	private double stdev[], average[]; //standard deviation and average of each attribute
	private int instancesCl[];
	private double frecuentCl[];

	private ArrayList <String>  clasesArray;

	/**
	 * Init a new set of instances
	 */
	public myDataset() {
		IS = new InstanceSet();
	}

	public myDataset clone(){

		myDataset copy= new myDataset();
		copy.IS = this.IS; 
		copy.X = new double[X.length][X[0].length];
		for (int i = 0; i < X.length; i++){
			copy.X[i] = this.X[i].clone();
		}
		copy.missing = new boolean[missing.length][missing[0].length];
		for (int i = 0; i < missing.length; i++){
			copy.missing[i] = this.missing[i].clone();
		}

		copy.nominal = this.nominal.clone();
		copy.instancesCl = this.instancesCl.clone();
		copy.integer = this.integer.clone();
		copy.output = this.output.clone();
		copy.outputInteger = this.outputInteger.clone();
		copy.outputReal = this.outputReal.clone();
		copy.emax = this.emax.clone();
		copy.emin = this.emin.clone();
		copy.average = this.average.clone();
		copy.stdev = this.stdev.clone();
		copy.instancesCl = this.instancesCl.clone();
		copy.frecuentCl = this.frecuentCl;

		copy.clasesArray = new ArrayList <String>();
		copy.clasesArray.addAll(clasesArray);

		copy.nClasses = this.nClasses;
		copy.nData = this.nData;
		copy.nInputs = this.nInputs;
		copy.nVars = this.nVars;

		return copy;

	}

	/**
	 * Outputs an array of examples with their corresponding attribute values.
	 * @return double[][] an array of examples with their corresponding attribute values
	 */
	public double[][] getX() {
		return X;
	}

	/**
	 * Output a specific example
	 * @param pos int position (id) of the example in the data-set
	 * @return double[] the attributes of the given example
	 */
	public double[] getExample(int pos) {
		return X[pos];
	}

	/**
	 * Returns the output of the data-set as integer values
	 * @return int[] an array of integer values corresponding to the output values of the dataset
	 */
	public int[] getOutputAsInteger() {
		int[] output = new int[outputInteger.length];
		for (int i = 0; i < outputInteger.length; i++) {
			output[i] = outputInteger[i];
		}
		return output;
	}

	/**
	 * Returns the output of the data-set as real values
	 * @return double[] an array of real values corresponding to the output values of the dataset
	 */
	public double[] getOutputAsReal() {
		double[] output = new double[outputReal.length];
		for (int i = 0; i < outputReal.length; i++) {
			output[i] = outputInteger[i];
		}
		return output;
	}

	/**
	 * Returns the output of the data-set as nominal values
	 * @return String[] an array of nominal values corresponding to the output values of the dataset
	 */
	public String[] getOutputAsString() {
		String[] output = new String[this.output.length];
		for (int i = 0; i < this.output.length; i++) {
			output[i] = this.output[i];
		}
		return output;
	}

	/**
	 * It returns the output value of the example "pos"
	 * @param pos int the position (id) of the example
	 * @return String a string containing the output value
	 */
	public String getOutputAsString(int pos) {
		return output[pos];
	}

	/**
	 * It returns the output value of the example "pos"
	 * @param pos int the position (id) of the example
	 * @return int an integer containing the output value
	 */
	public int getOutputAsInteger(int pos) {
		return outputInteger[pos];
	}

	/**
	 * It returns the output value of the example "pos"
	 * @param pos int the position (id) of the example
	 * @return double a real containing the output value
	 */
	public double getOutputAsReal(int pos) {
		return outputReal[pos];
	}

	/**
	 * It returns an array with the maximum values of the attributes
	 * @return double[] an array with the maximum values of the attributes
	 */
	public double[] getemax() {
		return emax;
	}

	/**
	 * It returns an array with the minimum values of the attributes
	 * @return double[] an array with the minimum values of the attributes
	 */
	public double[] getemin() {
		return emin;
	}

	/**
	 * It returns an the maximum of a given attribute
	 * @return double the maximum value of the attribute
	 */
	public double getMax(int variable) {
		return emax[variable];
	}

	/**
	 * It returns an the minimum of a given attribute
	 * @return double the minimum value of the attribute
	 */
	public double getMin(int variable) {
		return emin[variable];
	}

	/**
	 * It gets the size of the data-set
	 * @return int the number of examples in the data-set
	 */
	public int getnData() {
		return nData;
	}

	/**
	 * It gets the number of variables of the data-set (including the output)
	 * @return int the number of variables of the data-set (including the output)
	 */
	public int getnVars() {
		return nVars;
	}

	/**
	 * It gets the number of input attributes of the data-set
	 * @return int the number of input attributes of the data-set
	 */
	public int getnInputs() {
		return nInputs;
	}

	/**
	 * It gets the number of output attributes of the data-set (for example number of classes in classification)
	 * @return int the number of different output values of the data-set
	 */
	public int getnClasses() {
		return nClasses;
	}

	/**
	 * This function checks if the attribute value is missing
	 * @param i int Example id
	 * @param j int Variable id
	 * @return boolean True is the value is missing, else it returns false
	 */
	public boolean isMissing(int i, int j) {
		return missing[i][j];
	}

	/**
	 * This function checks if the attribute value is nominal
	 * @param i int attribute id
	 * @return boolean True is the value is nominal, else it returns false
	 */
	public boolean isNominal(int i) {
		return nominal[i];
	}

	/**
	 * This function checks if the attribute value is integer
	 * @param i int attribute id
	 * @return boolean True is the value is integer, else it returns false
	 */
	public boolean isInteger(int i) {
		return integer[i];
	}

	/**
	 * It reads the whole input data-set and it stores each example and its associated output value in
	 * local arrays to ease their use.
	 * @param datasetFile String name of the file containing the dataset
	 * @param train boolean It must have the value "true" if we are reading the training data-set
	 * @throws IOException If there ocurs any problem with the reading of the data-set
	 */
	public void readClassificationSet(String datasetFile, boolean train) throws IOException {
		try {
			// Load in memory a dataset that contains a classification problem
			IS.readSet(datasetFile, train);
			nData = IS.getNumInstances();
			nInputs = Attributes.getInputNumAttributes();
			nVars = nInputs + Attributes.getOutputNumAttributes();

			// outputIntegerheck that there is only one output variable
			if (Attributes.getOutputNumAttributes() > 1) {
				System.out.println(
						"This algorithm can not process MIMO datasets");
				System.out.println(
				"All outputs but the first one will be removed");
				System.exit(1);
			}
			boolean noOutputs = false;
			if (Attributes.getOutputNumAttributes() < 1) {
				System.out.println(
						"This algorithm can not process datasets without outputs");
				System.out.println("Zero-valued output generated");
				noOutputs = true;
				System.exit(1);
			}

			// Initialice and fill our own tables
			X = new double[nData][nInputs];
			missing = new boolean[nData][nInputs];
			nominal = new boolean[nInputs];
			integer = new boolean[nInputs];
			outputInteger = new int[nData];
			outputReal = new double[nData];
			output = new String[nData];

			// Maximum and minimum of inputs
			emax = new double[nInputs];      
			emin = new double[nInputs];
			for (int i = 0; i < nInputs; i++) {
				if (Attributes.getInputAttribute(i).getNumNominalValues() > 0) {
					emin[i] = 0;
					emax[i] = Attributes.getInputAttribute(i).getNumNominalValues() - 1;
				}
				else {
					emin[i] = Attributes.getInputAttribute(i).getMinAttribute();
					emax[i] = Attributes.getInputAttribute(i).getMaxAttribute();
				}
				if (Attributes.getInputAttribute(i).getType() == Attribute.NOMINAL) {
					nominal[i] = true;
					integer[i] = false;
				}
				else if (Attributes.getInputAttribute(i).getType() == Attribute.INTEGER) {
					nominal[i] = false;
					integer[i] = true;
				}
				else {
					nominal[i] = false;
					integer[i] = false;
				}
			}

			// All values are casted into double/integer
			nClasses = 0;
			for (int i = 0; i < nData; i++) {
				Instance inst = IS.getInstance(i);
				for (int j = 0; j < nInputs; j++) {
					X[i][j] = IS.getInputNumericValue(i, j); //inst.getInputRealValues(j);
					missing[i][j] = inst.getInputMissingValues(j);
					if (missing[i][j])  X[i][j] = emin[j]-1;
				}

				if (noOutputs) {
					outputInteger[i] = 0;
					output[i] = "";
				} else {
					outputInteger[i] = (int) IS.getOutputNumericValue(i, 0);
					output[i] = IS.getOutputNominalValue(i, 0);
				}
				if (outputInteger[i] > nClasses) {
					nClasses = outputInteger[i];
				}
			}
			nClasses++;
			System.out.println("Number of classes=" + nClasses);

		} catch (Exception e) {
			System.out.println("DBG: Exception in readSet");
			e.printStackTrace();
		}

		Vector <String> auxiliar = Attributes.getOutputAttribute(0).getNominalValuesList();
		this.clasesArray= new ArrayList <String>();
		clasesArray.addAll(auxiliar);

		this.computeInstancesPerClass();
	}


	/**
	 * It copies the header of the dataset
	 * @return String A string containing all the data-set information
	 */
	public String copyHeader() {
		String p = new String("");
		p = "@relation " + Attributes.getRelationName() + "\n";
		p += Attributes.getInputAttributesHeader();
		p += Attributes.getOutputAttributesHeader();
		p += Attributes.getInputHeader() + "\n";
		p += Attributes.getOutputHeader() + "\n";
		p += "@data\n";
		return p;
	}

	/**
	 * It checks if the data-set has any real value
	 * @return boolean True if it has some real values, else false.
	 */
	public boolean hasRealAttributes() {
		return Attributes.hasRealAttributes();
	}

	/**
	 * It checks if the data-set has any integer value
	 * @return boolean True if it has some integer values, else false.
	 */
	public boolean hasNumericalAttributes() {
		return (Attributes.hasIntegerAttributes() || Attributes.hasRealAttributes());
	}

	/**
	 * It checks if the data-set has any missing value
	 * @return boolean True if it has some missing values, else false.
	 */
	public boolean hasMissingAttributes() {
		return (this.sizeWithoutMissing() < this.getnData());
	}

	/**
	 * It return the size of the data-set without having account the missing values
	 * @return int the size of the data-set without having account the missing values
	 */
	public int sizeWithoutMissing() {
		int i, j, tam;

		tam = 0;
		for (i = 0; i < nData; i++) {
			for (j = 1; (j < nInputs) && (!isMissing(i, j)); j++);
			if (j == nInputs)  tam++;
		}

		return tam;
	}

	/**
	 * It returns the total amount of instances
	 * @return the size of the dataset
	 */
	public int size() {
		return nData;
	}

	public int getMajority(){
		int index = 0;
		for (int i = 1; i < this.instancesCl.length; i++){
			if (instancesCl[i] > instancesCl[index]){    			
				index = i;
			}
		}
		return instancesCl[index];
	}

	/**
	 * It computes the number of instances per class to stroe it in "instacesCl" array
	 */
	public void computeInstancesPerClass() {
		int i;
		this.instancesCl = new int[this.nClasses];
		this.frecuentCl = new double[this.nClasses];

		for (i = 0; i < this.nClasses; i++)  this.instancesCl[i] = 0;
		for (i = 0; i < this.getnData(); i++)  this.instancesCl[this.outputInteger[i]]++;
		for (i = 0; i < this.nClasses; i++)  this.frecuentCl[i] = (1.0 * this.instancesCl[i]) / (double) this.nData;
	}

	/**
	 * It computes the number of instances for a given class
	 * @param clas id of the class
	 * @return the number of examples with the former class lable
	 */
	public int numberInstances(int clas) {
		return this.instancesCl[clas];
	}

	/**
	 * It returns the class distribution
	 * @param clas id of the class
	 * @return the a priori class distribution 
	 */
	public double frecuentClass(int clas) {
		return frecuentCl[clas];
	}

	/**
	 * It gets the number of nominal values for a given attribute
	 * @param attribute id of the variable
	 * @return the number of nominal values for a given attribute
	 */
	public int numberValues(int attribute) {
		return Attributes.getInputAttribute(attribute).getNumNominalValues();
	}

	/**
	 * It gets a class label
	 * @param intValue id of the class
	 * @return class label (string)
	 */
	public String getOutputValue(int intValue) {
		return Attributes.getOutputAttribute(0).getNominalValue(intValue);
	}

	/**
	 * It outputs the type of the variable
	 * @param variable the id of the variable (position)
	 * @return the type: INTEGER, REAL, OR NOMINAL
	 */
	public int getType(int variable) {
		if (Attributes.getAttribute(variable).getType() == Attributes.getAttribute(0).INTEGER)   return this.INTEGER;
		if (Attributes.getAttribute(variable).getType() == Attributes.getAttribute(0).REAL)  return this.REAL;
		if (Attributes.getAttribute(variable).getType() == Attributes.getAttribute(0).NOMINAL)  return this.NOMINAL;

		return 0;
	}

	/**
	 * It returns the universe of discourse of the input and output variables
	 * @return double[][] the minimum and maximum values per variable
	 */
	public double [][] returnRanks(){
		double [][] ranges = new double[this.getnVars()][2];
		for (int i = 0; i < this.getnInputs(); i++){
			if (Attributes.getInputAttribute(i).getNumNominalValues() > 0){
				ranges[i][0] = 0;
				ranges[i][1] = Attributes.getInputAttribute(i).getNumNominalValues()-1;
			}
			else{
				ranges[i][0] = Attributes.getInputAttribute(i).getMinAttribute();
				ranges[i][1] = Attributes.getInputAttribute(i).getMaxAttribute();
			}
		}
		ranges[this.getnVars()-1][0] = Attributes.getOutputAttribute(0).getMinAttribute();
		ranges[this.getnVars()-1][1] = Attributes.getOutputAttribute(0).getMaxAttribute();
		return ranges;
	}


	public String [] names(){
		String names[] = new String[nInputs];
		for (int i = 0; i < nInputs; i++){
			names[i] = Attributes.getInputAttribute(i).getName();
		}
		return names;
	}

	public String [] clases(){
		String clases[] = new String[nClasses];
		for (int i = 0; i < nClasses; i++){
			clases[i] = Attributes.getOutputAttribute(0).getNominalValue(i);
		}
		return clases;
	}

	public ArrayList <String> getClases(){
		return clasesArray;
	}

	public static double realValue(int atributo, String valorNominal){
		Vector nominales = Attributes.getInputAttribute(atributo).getNominalValuesList();
		int aux = nominales.indexOf(valorNominal);
		return 1.0*aux;
	}

	public int numericClass(String valorNominal){
		Vector nominales = Attributes.getOutputAttribute(0).getNominalValuesList();
		int aux = nominales.indexOf(valorNominal);
		return aux;
	}

	public static String nominalValue(int atributo, double valorReal){
		Vector nominales = Attributes.getInputAttribute(atributo).getNominalValuesList();
		return (String)nominales.get((int)valorReal);
	}

	/**
	 * It returns the number of nominal values for a given variable
	 * @param attribute var id
	 * @return the number of nominal values for a given variable
	 */
	public int totalNominals(int attribute){
		return Attributes.getInputAttribute(attribute).getNumNominalValues();
	}

	/**
	 * It computes the most frequent class in the dataset
	 * @return the label of the most frequent class
	 */
	public String mostFrequentClass(){
		int index = 0;
		for (int i = 1; i < this.instancesCl.length; i++){
			if (instancesCl[i] > instancesCl[index]){    			
				index = i;
			}
		}
		return this.clases()[index];
	}

}
