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

import java.util.Enumeration;
import wrapper.myDataset;

import java.util.Vector;
import keel.Dataset.Attributes;
import keel.Dataset.DatasetException;
import keel.Dataset.HeaderFormatException;
import keel.Dataset.Instance;
import keel.Dataset.InstanceSet;


/**
 * Class to implement the dataset
 * 
 * <p>
 * @author Written by Cristobal Romero (Universidad de C�rdoba) 10/10/2007
 * @version 0.1
 * @since JDK 1.5
 *</p>
 */
public class Dataset {
	/** The name of the dataset. */
	protected String name = "";

	/** The attributes. */
	protected Vector attributes;

	/** The itemsets. */
	protected Vector itemsets;

	/** The index of the class attribute. */
	protected int classIndex;

	/** Keel dataset InstanceSet **/
	protected InstanceSet IS;

	protected boolean [] minority;
	
	protected int [] n_classes;

	/** Function to read the .dat file that contains the information of the dataset.
	 *
	 * @param name 		The reader object where the itemsets are readed.
	 * @param variables array with selected variables
	 * @param train     The flag if the file is for training
	 */
	public Dataset(String name, boolean [] variables, boolean [] ejemplos, boolean train) {
		try {
			// create the set of instances
			IS = new InstanceSet();
			// Read the itemsets.
			IS.readSet(name, train);
		} catch (DatasetException e) {
			System.out.println("Error loading dataset instances");
			e.printStackTrace();
			System.exit( -1);
		} catch (HeaderFormatException e) {
			System.out.println("Error loading dataset instances");
			e.printStackTrace();
			System.exit( -1);
		}

		//computeMinorityClass();

		//Store Dataset file attributes
		readHeader(variables);

		//itemsets = new Vector(IS.getNumInstances());
		itemsets = new Vector(ejemplos.length);

		// read all the itemsets
		getItemsetFull(variables,ejemplos);

	}

	/** Function to read the .dat file that contains the information of the dataset.
	 *
	 * @param name 		The reader object where the itemsets are readed.
	 * @param variables array with selected variables
	 * @param train     The flag if the file is for training
	 */
	public Dataset(myDataset train, boolean [] variables, boolean [] ejemplos) {

		// create the set of instances
		IS = new InstanceSet();
		// Read the itemsets.
		IS = train.IS;

		//Store Dataset file attributes
		readHeader(variables);

		//itemsets = new Vector(IS.getNumInstances());
		itemsets = new Vector(ejemplos.length);

		// read all the itemsets
		getItemsetFull(variables,ejemplos);
		
		computeNumberClasses();

	}
	
	/** Function to read the .dat file that contains the information of the dataset.
	 *
	 * @param name 		The reader object where the itemsets are readed.
	 * @param variables array with selected variables
	 * @param train     The flag if the file is for training
	 */
	public Dataset(myDataset train, boolean [] variables) {
		IS = train.IS;

		//Store Dataset file attributes
		readHeader(variables);

		itemsets = new Vector(IS.getNumInstances());

		// read all the itemsets
		getItemsetFull(variables);
		
		computeNumberClasses();

	}
	
	private void computeNumberClasses(){
		this.n_classes = new int[this.numClasses()];
		for (int i = 0; i < this.itemsets.size(); i++){
			Itemset itemset = this.itemset(i);
			n_classes[(int)itemset.getValue(this.getClassIndex())]++;
		}
	}
	
	public int size(int index){
		return n_classes[index];
	}

	/** Function to read the .dat file that contains the information of the dataset.
	 *
	 * @param name 		The reader object where the itemsets are readed.
	 * @param variables array with selected variables
	 * @param train     The flag if the file is for training
	 */
	public Dataset(String name, boolean [] variables, boolean train) {
		try {
			// create the set of instances
			IS = new InstanceSet();
			// Read the itemsets.
			IS.readSet(name, train);
		} catch (DatasetException e) {
			System.out.println("Error loading dataset instances");
			e.printStackTrace();
			System.exit( -1);
		} catch (HeaderFormatException e) {
			System.out.println("Error loading dataset instances");
			e.printStackTrace();
			System.exit( -1);
		}

		//Store Dataset file attributes
		readHeader(variables);

		itemsets = new Vector(IS.getNumInstances());

		// read all the itemsets
		getItemsetFull(variables);

	}

	/** Function to read the .dat file that contains the information of the dataset.
	 *
	 * @param name 		The reader object where the itemsets are readed.
	 * @param variables array with selected variables
	 * @param train     The flag if the file is for training
	 */
	public Dataset(String name, boolean train) {
		try {
			// create the set of instances
			IS = new InstanceSet();
			// Read the itemsets.
			IS.readSet(name, train);
		} catch (DatasetException e) {
			System.out.println("Error loading dataset instances");
			e.printStackTrace();
			System.exit( -1);
		} catch (HeaderFormatException e) {
			System.out.println("Error loading dataset instances");
			e.printStackTrace();
			System.exit( -1);
		}

		//Store Dataset file attributes
		readHeader();

		itemsets = new Vector(IS.getNumInstances());

		// read all the itemsets
		getItemsetFull();

	}

	/** Constructor that copies another dataset.
	 *
	 * @param dataset		The dataset to be copied.
	 */
	public Dataset(Dataset dataset) {
		this(dataset, dataset.numItemsets());
		dataset.copyItemsets(0, this, dataset.numItemsets());
	}


	/** Constructor to copy all the attributes of another dataset but the itemsets.
	 *
	 * @param dataset		The dataset to be copied.
	 * @param capacity		The number of itemsets.
	 */
	public Dataset(Dataset dataset, int capacity) {
		if (capacity < 0) {
			capacity = 0;
		}

		classIndex = dataset.classIndex;
		name = dataset.getName();
		attributes = dataset.attributes;
		itemsets = new Vector(capacity);
	}

	private void computeMinorityClass(){    	
		int [] countClass = new int[2];
		for (int i = 0; i < IS.getNumInstances(); i++) {
			countClass[(int) IS.getOutputNumericValue(i, 0)]++;
		}
		int min = 0;
		if (countClass[0] > countClass[1]){
			min = 1;
		}
		minority = new boolean[IS.getNumInstances()];   	                        
		for(int i = 0; i < IS.getNumInstances(); i++){
			minority[i] = ((int) IS.getOutputNumericValue(i, 0) == min);
		}
	}

	/** 
	 * Function to stores header of a data file.
	 */
	private void readHeader(boolean [] variables) {
		String attributeName;
		Vector attributeValues;

		name = Attributes.getRelationName();

		// Create vectors to hold information temporarily.
		attributes = new Vector();

		keel.Dataset.Attribute at;

		// store attribute inputs and of the header
		for (int i = 0; i < variables.length; i++) {
			if (variables[i]){
				//System.out.println("Leo la variable: "+i+"/"+Attributes.getInputNumAttributes());
				at = Attributes.getInputAttribute(i);
				attributeName = at.getName();
				// check if it is real
				if (at.getType() == 2) {
					float min = (float) at.getMinAttribute();
					float max = (float) at.getMinAttribute();
					attributes.addElement(new Attribute(attributeName, i));
					Attribute att = (Attribute) attributes.lastElement(); // attributes.elementAt(i);
					att.setRange(min, max);
					att.activate();
				} else {
					if (at.getType() == 1) { // check if it is integer
						int min = (int) at.getMinAttribute();
						int max = (int) at.getMinAttribute();
						attributes.addElement(new Attribute(attributeName, i));
						Attribute att = (Attribute) attributes.lastElement(); // elementAt(i);
						att.setRange(min, max);
						att.activate();
					} else { // it is nominal
						attributeValues = new Vector();
						for (int k = 0; k < at.getNumNominalValues(); k++) {
							attributeValues.addElement(at.getNominalValue(k));
						}
						attributes.addElement(new Attribute(attributeName, attributeValues, i));
						Attribute att = (Attribute) attributes.lastElement(); // .elementAt(i);
						att.activate();
					}
				}
			}//if variables
		} //for
		
		// store outputs of the header
		at = Attributes.getOutputAttribute(0);
		attributeName = at.getName();
		int j = variables.length; //Attributes.getNumAttributes() - 1;
		// check if it is real
		if (at.getType() == 2) {
			float min = (float) at.getMinAttribute();
			float max = (float) at.getMinAttribute();
			attributes.addElement(new Attribute(attributeName, j));
			Attribute att = (Attribute) attributes.lastElement(); // .elementAt(j);
			att.setRange(min, max);
			att.activate();
		} else {
			if (at.getType() == 1) { // check if it is integer
				int min = (int) at.getMinAttribute();
				int max = (int) at.getMinAttribute();
				attributes.addElement(new Attribute(attributeName, j));
				Attribute att = (Attribute) attributes.lastElement(); // .elementAt(j);
				att.setRange(min, max);
				att.activate();
			} else { // it is nominal
				attributeValues = new Vector();
				for (int k = 0; k < at.getNumNominalValues(); k++) {
					attributeValues.addElement(at.getNominalValue(k));
				}
				attributes.addElement(new Attribute(attributeName,attributeValues, j));
				Attribute att = (Attribute) attributes.lastElement(); // elementAt(j);
				att.activate();
			}
		}

		// set the index of the output class
		classIndex = attributes.size()-1;
		
		//System.err.println("Clase: "+classIndex);
		//System.err.println("Ejemplo: "+((Attribute)attributes.get(0)).name());
	}

	/** 
	 * Function to stores header of a data file.
	 */
	private void readHeader() {
		String attributeName;
		Vector attributeValues;
		name = Attributes.getRelationName();

		// Create vectors to hold information temporarily.
		attributes = new Vector();

		keel.Dataset.Attribute at;
		// store attribute inputs and of the header
		for (int j = 0; j < Attributes.getInputNumAttributes(); j++) {
			at = Attributes.getInputAttribute(j);
			attributeName = at.getName();
			// check if it is real
			if (at.getType() == 2) {
				float min = (float) at.getMinAttribute();
				float max = (float) at.getMinAttribute();
				attributes.addElement(new Attribute(attributeName, j));
				Attribute att = (Attribute) attributes.elementAt(j);
				att.setRange(min, max);
				att.activate();
			} else {
				if (at.getType() == 1) { // check if it is integer
					int min = (int) at.getMinAttribute();
					int max = (int) at.getMinAttribute();
					attributes.addElement(new Attribute(attributeName, j));
					Attribute att = (Attribute) attributes.elementAt(j);
					att.setRange(min, max);
					att.activate();
				} else { // it is nominal
					attributeValues = new Vector();
					for (int k = 0; k < at.getNumNominalValues(); k++) {
						attributeValues.addElement(at.getNominalValue(k));
					}
					attributes.addElement(new Attribute(attributeName, attributeValues, j));
					Attribute att = (Attribute) attributes.elementAt(j);
					att.activate();
				}
			}
		} //for

		// store outputs of the header
		at = Attributes.getOutputAttribute(0);
		attributeName = at.getName();
		int j = Attributes.getNumAttributes() - 1;
		// check if it is real
		if (at.getType() == 2) {
			float min = (float) at.getMinAttribute();
			float max = (float) at.getMinAttribute();
			attributes.addElement(new Attribute(attributeName, j));
			Attribute att = (Attribute) attributes.elementAt(j);
			att.setRange(min, max);
			att.activate();
		} else {
			if (at.getType() == 1) { // check if it is integer
				int min = (int) at.getMinAttribute();
				int max = (int) at.getMinAttribute();
				attributes.addElement(new Attribute(attributeName, j));
				Attribute att = (Attribute) attributes.elementAt(j);
				att.setRange(min, max);
				att.activate();
			} else { // it is nominal
				attributeValues = new Vector();
				for (int k = 0; k < at.getNumNominalValues(); k++) {
					attributeValues.addElement(at.getNominalValue(k));
				}
				attributes.addElement(new Attribute(attributeName,attributeValues, j));
				Attribute att = (Attribute) attributes.elementAt(j);
				att.activate();
			}
		}

		// set the index of the output class
		classIndex = Attributes.getNumAttributes() - 1;
	}


	/** Function to read an itemset and appends it to the dataset.
	 *
	 *
	 * @return					True if the itemset was readed succesfully.
	 *
	 */
	private boolean getItemsetFull(boolean [] variables, boolean [] ejemplosIni) {
		//fill itemset
		boolean [] ejemplos = new boolean [IS.getNumInstances()];
		if (ejemplosIni.length < IS.getNumInstances()){ //only majority class
			for (int j = 0,jj = 0; j < IS.getNumInstances(); j++) {
				if (IS.getInstance(j).getOutputNominalValues(0).equalsIgnoreCase("positive")){
					ejemplos[j] = true;
				}else{
					ejemplos[j] = ejemplosIni[jj++];
				}
			}
		}else{
			ejemplos = ejemplosIni.clone();
		}
		for (int j = 0; j < IS.getNumInstances(); j++) {
			if (ejemplos[j]){
				double[] itemset = new double[attributes.size()]; //?
				// Get values for all input attributes.
				for (int i = 0, ii = 0; i < variables.length; i++) {
					// check type and if there is null
					if (variables[i]){
						if (IS.getInstance(j).getInputMissingValues(i)) {
							itemset[ii] = Itemset.getMissingValue();
						} else {
							if (Attributes.getInputAttribute(i).getType() == 0) { //nominal
								for (int k = 0; k < Attributes.getInputAttribute(i).getNumNominalValues(); k++) {
									if (Attributes.getInputAttribute(i).getNominalValue(k).equals(IS.getInstance(j).
											getInputNominalValues(i))) {
										itemset[ii] = (double) k;
									}
								}
							} else { // real and integer
								itemset[ii] = IS.getInstance(j).getInputRealValues(i);
							}
						} // else
						ii++;
					}//if variables
				} //for

				// Get values for output attribute.
				//int i = variables.length; //Attributes.getInputNumAttributes();
				int i = attributes.size()-1;
				//check type and if there is null
				if (IS.getInstance(j).getOutputMissingValues(0)) {
					itemset[i] = Itemset.getMissingValue();
				} else {
					if (Attributes.getOutputAttribute(0).getType() == 0) { //nominal
						for (int k = 0; k < Attributes.getOutputAttribute(0).getNumNominalValues(); k++) {
							if (Attributes.getOutputAttribute(0).getNominalValue(k).
									equals(IS.getInstance(j).getOutputNominalValues(0))) {
								itemset[i] = (double) k;
							}
						}
					} else { // real and integer
						itemset[i] = IS.getInstance(j).getOutputRealValues(0);
					}
				} // else
				// Add itemset to dataset
				addItemset(new Itemset(1, itemset));
			}
		} // for
		return true;
	}

	/** Function to read an itemset and appends it to the dataset.
	 *
	 *
	 * @return					True if the itemset was readed succesfully.
	 *
	 */
	private boolean getItemsetFull(boolean [] variables) {
		//fill itemset
		for (int j = 0; j < IS.getNumInstances(); j++) {
			double[] itemset = new double[attributes.size()];
			// Get values for all input attributes.
			for (int i = 0, ii = 0; i < variables.length; i++) {
				// check type and if there is null
				if (variables[i]){
					if (IS.getInstance(j).getInputMissingValues(i)) {
						itemset[ii] = Itemset.getMissingValue();
					} else {
						if (Attributes.getInputAttribute(i).getType() == 0) { //nominal
							for (int k = 0; k < Attributes.getInputAttribute(i).getNumNominalValues(); k++) {
								if (Attributes.getInputAttribute(i).getNominalValue(k).equals(IS.getInstance(j).
										getInputNominalValues(i))) {
									itemset[ii] = (double) k;
								}
							}
						} else { // real and integer
							itemset[ii] = IS.getInstance(j).getInputRealValues(i);
						}
					} // else
					ii++;
				}//if variables
			} //for

			// Get values for output attribute.
			int i = attributes.size()-1; //Attributes.getInputNumAttributes();
			//check type and if there is null
			if (IS.getInstance(j).getOutputMissingValues(0)) {
				itemset[i] = Itemset.getMissingValue();
			} else {
				if (Attributes.getOutputAttribute(0).getType() == 0) { //nominal
					for (int k = 0; k < Attributes.getOutputAttribute(0).getNumNominalValues(); k++) {
						if (Attributes.getOutputAttribute(0).getNominalValue(k).
								equals(IS.getInstance(j).getOutputNominalValues(0))) {
							itemset[i] = (double) k;
						}
					}
				} else { // real and integer
					itemset[i] = IS.getInstance(j).getOutputRealValues(0);
				}
			} // else
			// Add itemset to dataset
			addItemset(new Itemset(1, itemset));
		} // for
		return true;
	}

	/** Function to read an itemset and appends it to the dataset.
	 *
	 *
	 * @return					True if the itemset was readed succesfully.
	 *
	 */
	private boolean getItemsetFull() {
		//fill itemset
		for (int j = 0; j < IS.getNumInstances(); j++) {
			double[] itemset = new double[Attributes.getNumAttributes()];
			int index;
			// Get values for all input attributes.
			for (int i = 0; i < Attributes.getInputNumAttributes(); i++) {
				// check type and if there is null
				if (IS.getInstance(j).getInputMissingValues(i)) {
					itemset[i] = Itemset.getMissingValue();
				} else {
					if (Attributes.getInputAttribute(i).getType() == 0) { //nominal
						for (int k = 0; k < Attributes.getInputAttribute(i).getNumNominalValues(); k++) {
							if (Attributes.getInputAttribute(i).getNominalValue(k).equals(IS.getInstance(j).
									getInputNominalValues(i))) {
								itemset[i] = (double) k;
							}
						}
					} else { // real and integer
						itemset[i] = IS.getInstance(j).getInputRealValues(i);
					}
				} // else
			} //for

			// Get values for output attribute.
			int i = Attributes.getInputNumAttributes();
			//check type and if there is null
			if (IS.getInstance(j).getOutputMissingValues(0)) {
				itemset[i] = Itemset.getMissingValue();
			} else {
				if (Attributes.getOutputAttribute(0).getType() == 0) { //nominal
					for (int k = 0; k < Attributes.getOutputAttribute(0).getNumNominalValues(); k++) {
						if (Attributes.getOutputAttribute(0).getNominalValue(k).
								equals(IS.getInstance(j).getOutputNominalValues(0))) {
							itemset[i] = (double) k;
						}
					}
				} else { // real and integer
					itemset[i] = IS.getInstance(j).getOutputRealValues(0);
				}
			} // else
			// Add itemset to dataset
			addItemset(new Itemset(1, itemset));
		} // for
		return true;
	}


	/** Function to add one itemset.
	 *
	 * @param itemset	The itemset to add to the dataset.
	 */
	public final void addItemset(Itemset itemset) {
		Itemset newItemset = (Itemset) itemset.copy();

		newItemset.setDataset(this);
		itemsets.addElement(newItemset);

	}

	/** Returns the name of the dataset.
	 *
	 */
	public String getName() {
		return name;
	}

	/** Returns the attribute that has the index.
	 *
	 * @param index		The index of the attribute.
	 */
	public final Attribute getAttribute(int index) {
		return (Attribute) attributes.elementAt(index);
	}

	/** Returns the attribute that has the name.
	 *
	 * @param name	The name of the attribute.
	 */
	public final Attribute getAttribute(String name) {
		for (int i = 0; i < attributes.size(); i++) {
			if (((Attribute) attributes.elementAt(i)).name().equalsIgnoreCase(
					name)) {
				return (Attribute) attributes.elementAt(i);
			}
		}

		return null;
	}

	/** Returns class attribute.
	 *
	 */
	public final Attribute getClassAttribute() {
		if (classIndex < 0) {
			System.err.println("Class index wrong:" + classIndex);
			return null;
		}
		return getAttribute(classIndex);
	}

	/** Returns the index of the class attribute.
	 *
	 */
	public final int getClassIndex() {
		return classIndex;
	}

	/** Returns the number of attributes.
	 *
	 */
	public final int numAttributes() {
		return attributes.size();
	}

	/** Returns the number of possible values of the class attribute.
	 *
	 */
	public final int numClasses() {
		if (classIndex < 0) {
			System.err.println("Class index wrong:" + classIndex);
			return -1;
		}
		return getClassAttribute().numValues();
	}

	/** Returns the number of itemsets.
	 *
	 */
	public final int numItemsets() {
		return itemsets.size();
	}

	/** Function to remove an itemset at the given position.
	 *
	 * @param index 	The index of the itemset to be deleted.
	 */
	public final void delete(int index) {
		itemsets.removeElementAt(index);
	}

	/** Function to remove all the attributes with missing value in the given attribute.
	 *
	 * @param attIndex		The index of the attribute.
	 */
	public final void deleteWithMissing(int attIndex) {
		Vector newItemsets = new Vector(numItemsets());

		for (int i = 0; i < numItemsets(); i++) {
			if (!itemset(i).isMissing(attIndex)) {
				newItemsets.addElement(itemset(i));
			}
		}

		itemsets = newItemsets;
	}

	/** Enumerates all the attributes.
	 *
	 * @return An enumeration that contains all the attributes.
	 */
	public Enumeration enumerateAttributes() {
		Vector help = new Vector(attributes.size() - 1);

		for (int i = 0; i < attributes.size(); i++) {
			if (i != classIndex) {
				help.addElement(attributes.elementAt(i));
			}
		}

		return help.elements();
	}

	/** Enumerates all the itemsets.
	 *
	 * @return	An enumeration that contains all the itemsets.
	 */
	public final Enumeration enumerateItemsets() {
		return itemsets.elements();
	}

	/** Returns the itemset at the given position.
	 *
	 * @param index	The index of the itemset.
	 */
	public final Itemset itemset(int index) {
		return (Itemset) itemsets.elementAt(index);
	}

	/** Returns the last itemset.
	 *
	 */
	public final Itemset lastItemset() {
		return (Itemset) itemsets.lastElement();
	}


	/** Function to add  the instances of one set to the end of another.
	 *
	 * @param from	The index of the first that is going to be copied.
	 * @param dest	The dataset where the itemsets are going to be copied.
	 * @param num	The number of itemsets to copy.
	 */
	private void copyItemsets(int from, Dataset dest, int num) {
		for (int i = 0; i < num; i++) {
			dest.addItemset(itemset(from + i));
		}
	}

	/** Function to compute the sum of all the weights of the itemsets.
	 *
	 * @return	The weight of all the itemsets.
	 */
	public final double sumOfWeights() {
		double sum = 0;

		for (int i = 0; i < numItemsets(); i++) {
			sum += itemset(i).getWeight();
		}

		return sum;
	}

	/** Function to sort the dataset based on an attribute.
	 *
	 * @param attIndex	The index of the attribute.
	 */
	public final void sort(int attIndex) {
		int i, j;

		// move all dataset with missing values to end
		j = numItemsets() - 1;
		i = 0;

		while (i <= j) {
			if (itemset(j).isMissing(attIndex)) {
				j--;
			} else {
				if (itemset(i).isMissing(attIndex)) {
					swap(i, j);
					j--;
				}

				i++;
			}
		}

		quickSort(attIndex, 0, j);
	}

	/** Function to implementate the quicksort method.
	 *
	 * @param attIndex		The index of the attribute used to sort the itemsets.
	 * @param lo0			Minimum value.
	 * @param hi0			Maximum value.
	 */
	private void quickSort(int attIndex, int lo0, int hi0) {
		int lo = lo0, hi = hi0;
		double mid, midPlus, midMinus;

		if (hi0 > lo0) {
			// Arbitrarily establishing partition element as the
			// midpoint of the array.
			mid = itemset((lo0 + hi0) / 2).getValue(attIndex);
			midPlus = mid + 1e-6;
			midMinus = mid - 1e-6;

			// loop through the array until indices cross
			while (lo <= hi) {
				// find the first element that is greater than or equal to
				// the partition element starting from the left Index.
				while ((itemset(lo).getValue(attIndex) < midMinus) && (lo < hi0)) {
					++lo;
				}

				// find an element that is smaller than or equal to
				// the partition element starting from the right Index.
				while ((itemset(hi).getValue(attIndex) > midPlus) && (hi > lo0)) {
					--hi;
				}

				// if the indexes have not crossed, swap
				if (lo <= hi) {
					swap(lo, hi);
					++lo;
					--hi;
				}
			}

			// If the right index has not reached the left side of array
			// must now sort the left partition.
			if (lo0 < hi) {
				quickSort(attIndex, lo0, hi);
			}

			// If the left index has not reached the right side of array
			// must now sort the right partition.
			if (lo < hi0) {
				quickSort(attIndex, lo, hi0);
			}
		}
	}

	/** Function to swap two itemsets.
	 *
	 * @param i		The first itemset.
	 * @param j		The second itemset.
	 */
	private void swap(int i, int j) {
		Object help = itemsets.elementAt(i);

		itemsets.insertElementAt(itemsets.elementAt(j), i);
		itemsets.removeElementAt(i + 1);
		itemsets.insertElementAt(help, j);
		itemsets.removeElementAt(j + 1);
	}
	
	public void setWeights(double [] priorProbabilities){
		int minority = (int) priorProbabilities[0];
		for (int i = 1; i < priorProbabilities.length; i++){
			if (minority > priorProbabilities[i]){
				minority = (int)priorProbabilities[i];
			}
		}
		double [] weights = new double[priorProbabilities.length];
		for (int i = 0; i < weights.length; i++){
			weights[i] = minority/priorProbabilities[i];
		}
		// Change the weights of the itemsets in the dataset
        for (int i = 0; i < numItemsets(); i++) {
        	double value = weights[(int)itemset(i).getClassValue()];
        	itemset(i).setWeight(value);
        }
	}


}

