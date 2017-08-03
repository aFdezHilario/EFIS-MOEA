package wrapper;

public class AUC implements Comparable<Object>{

	private double pPos;
	private int predictedClass, actualClass;
	
	
	/**
	 * Compares the predicted positive values of two instances for the ordering procedure
	 * @param a Object an AUC object
	 * @return int -1 if the current pPos is worst than the one that is compared, 1 for the contrary case and 0
	 * if they are equal.
	 */
	public int compareTo(Object a) {
		int retorno = 0;
		if ( ( (AUC) a).pPos < this.pPos) {
			retorno = -1;
		}
		if ( ( (AUC) a).pPos > this.pPos) {
			retorno = 1;
		}
		return retorno;
	}
	
	public AUC(double pPos, int predictedClass, int actualClass){
		this.pPos = pPos;
		this.predictedClass = predictedClass;
		this.actualClass = actualClass;
	}
	
	public int getPrCl(){
		return this.predictedClass;
	}
	
	public int getActCl(){
		return this.actualClass;
	}
	
	public double getPpos(){
		return this.pPos;
	}
	
}
