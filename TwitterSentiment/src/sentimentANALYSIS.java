/*
// Purpose		:	To implement twitter sentiment analysis for Real time big data challenge 1
//Owner			:	Group 4 Members
//Team Members	:	Srikar Reddy Mallareddygari
//					Pavan Kumar Bollaram

*/
// Import for java Library 

import java.io.*;

import java.util.*;


// Import Weka Libraries

import weka.classifiers.Classifier;

import weka.classifiers.bayes.NaiveBayes;

import weka.core.Attribute;

import weka.core.Instance;

import weka.core.Instances;

import weka.core.SparseInstance;

import weka.core.converters.CSVLoader;

import cmu.arktweetnlp.POSTagger;

import cmu.arktweetnlp.Token;

// import CSV Library

import com.csvreader.*;



public class sentimentANALYSIS {

    

    private ArrayList<String> featureWords;

    private ArrayList<Attribute> attributeList;

    private Instances inputDataset;

    private POSTagger posTagger;

    private Classifier classifier;

    private ArrayList<String> sentimentClassList;

    

    public sentimentANALYSIS()

    {

        attributeList = new ArrayList<>();

        posTagger = new POSTagger();

                

        initialize();

    }

    private void initialize()

    {

        ObjectInputStream ois = null;

        try {

            //reads the feature words list to a hashset

            ois = new ObjectInputStream(new FileInputStream("FeatureWordsList.dat"));

            featureWords = (ArrayList<String>) ois.readObject();

        } catch (Exception ex) {

            System.out.println("Exception in Deserialization");

        } finally {

            try {

                ois.close();

            } catch (IOException ex) {

                System.out.println("Exception while closing file after Deserialization");

            }

        }

        

        //creating an attribute Weka list from the list of feature words given in Weka Tutorial

        sentimentClassList = new ArrayList<>();

        sentimentClassList.add("positive");

        sentimentClassList.add("negative");

        for(String featureWord : featureWords)

        {

            attributeList.add(new Attribute(featureWord));

        }

        
        attributeList.add(new Attribute("Sentiment",sentimentClassList));

    }

    

    

    public void trainClassifier(final String INPUT_FILENAME)

    {

            getTrainingDataset(INPUT_FILENAME);

            

            //trainingInstances consists of Weka feature vector of every input

            Instances trainingInstances = createInstances("TRAINING_INSTANCES");

            

            for(Instance currentInstance : inputDataset)

            {

                //******//extractFeature method gives the feature vector for the current I/P //*******

                Instance currentFeatureVector = extractFeature(currentInstance);

                currentFeatureVector.setDataset(trainingInstances);

                trainingInstances.add(currentFeatureVector);

            }

            

        //*****//NaiveBayes Classifier Tutorial //*****

        //********For instance classifier = new SMO; ************

        classifier = new NaiveBayes();

            

        try {

            //classifier training code

            classifier.buildClassifier(trainingInstances);

            

            //storing the trained classifier to a file for future use

            weka.core.SerializationHelper.write("NaiveBayes.model",classifier);

        } catch (Exception ex) {

            System.out.println("Exception in training the classifier.");

        }

    }

    

    

    public void testClassifier(final String INPUT_FILENAME)

    {

        getTrainingDataset(INPUT_FILENAME);

            

        //trainingInstances consists of feature vector of every input

        Instances testingInstances = createInstances("TESTING_INSTANCES");



        for(Instance currentInstance : inputDataset)

        {

            //extractFeature method returns the feature vector for the current input

            Instance currentFeatureVector = extractFeature(currentInstance);



            //Make the currentFeatureVector to be added to the trainingInstances

            currentFeatureVector.setDataset(testingInstances);

            testingInstances.add(currentFeatureVector);

        }

            

            

        try {

            //Classifier deserialization Weka

            classifier = (Classifier) weka.core.SerializationHelper.read("NaiveBayes.model");

            

            

            try {

         	   

    			File file = new File("/home/cloudera/Desktop/sentimenttemp.csv");

     

    			// If File not available creates a new file

    			if (!file.exists()) {

    				file.createNewFile();

    			}

     

    			FileWriter fw1 = new FileWriter(file.getAbsoluteFile(),true);

    			BufferedWriter bw1 = new BufferedWriter(fw1);

    		
    		     	bw1.write("sentiment");

         			bw1.write("\n");

         			bw1.close();

          


              }catch (IOException e) {

         			e.printStackTrace();

         		}

            

            //classifier testing code

            for(Instance testInstance : testingInstances)

            {

                double score = classifier.classifyInstance(testInstance);

              
                try {

              	   

        			File file = new File("/home/cloudera/Desktop/sentimenttemp.csv");

					//Writing data to a file using File Writer

        			if (!file.exists()) {

        				file.createNewFile();

        			}

         
        			FileWriter fw2 = new FileWriter(file.getAbsoluteFile(),true);

        			BufferedWriter bw2 = new BufferedWriter(fw2);

        			

        		      String content = testingInstances.attribute("Sentiment").value((int)score).toString();

             			bw2.write(content);

             			bw2.write("\n");

             			bw2.close();

                         		

                  }catch (IOException e) {

             			e.printStackTrace();

             		}

            } 

        }

            

         catch (Exception ex) {

            System.out.println("Exception in testing the classifier.");

        }

    }

    

    

    private void getTrainingDataset(final String INPUT_FILENAME)

    {

        try{

            //reading the training dataset from CSV file

            CSVLoader trainingLoader =new CSVLoader();

            trainingLoader.setSource(new File(INPUT_FILENAME));

            inputDataset = trainingLoader.getDataSet();

        }catch(IOException ex)

        {

            System.out.println("Exception in getTrainingDataset Method");

        }

    }

    

    

    private Instances createInstances(final String INSTANCES_NAME)

    {

        

        //create an Instances object with initial capacity as zero 

        Instances instances = new Instances(INSTANCES_NAME,attributeList,0);

        

        //sets the class index as the last attribute (positive or negative)

        instances.setClassIndex(instances.numAttributes()-1);

            

        return instances;

    }

    

    

    private Instance extractFeature(Instance inputInstance)

    {

        Map<Integer,Double> featureMap = new TreeMap<>();

        List<Token> tokens = posTagger.runPOSTagger(inputInstance.stringValue(0));



        for(Token token : tokens)

        {

            switch(token.getPOS())

            {

                case "A":

                case "V":

                case "R":   

                case "#":   

//                	System.out.println(token.getWord());

                	String word = token.getWord().replaceAll("#","");

                            if(featureWords.contains(word))

                            {

                                //adding 1.0 to the featureMap represents that the feature word is present in the input data

                                featureMap.put(featureWords.indexOf(word),1.0);

                            }

            }

        }

        int indices[] = new int[featureMap.size()+1];

        double values[] = new double[featureMap.size()+1];

        int i=0;

        for(Map.Entry<Integer,Double> entry : featureMap.entrySet())

        {

            indices[i] = entry.getKey();

            values[i] = entry.getValue();

            i++;

        }

        indices[i] = featureWords.size();

        values[i] = (double)sentimentClassList.indexOf(inputInstance.stringValue(1));

        return new SparseInstance(1.0,values,indices,featureWords.size());

    }

    

    

    public static void main(String[] args) throws Exception

    {

          sentimentANALYSIS wekaTutorial = new sentimentANALYSIS();

          wekaTutorial.trainClassifier("training.csv");

         //Added by Srikar for processed tweet output data from Source Project
          wekaTutorial.testClassifier("/home/cloudera/Desktop/tweetoutput.csv");

          // Concat of tweets and sentiment

          try {

     			

     			//Tweet file
        	  CsvReader tweet = new CsvReader("testing.csv");

     			//sentiment file
        	  CsvReader sentiment = new CsvReader("/home/cloudera/Desktop/sentimenttemp.csv");

     		
// Checking the files
     			tweet.readHeaders();

     			sentiment.readHeaders();



     			while (tweet.readRecord() && sentiment.readRecord())

     			{

     				String tweetID = tweet.get("Twitter");

     				tweet.get(0);

     				

     				String sentimentID = sentiment.get("sentiment");

     				sentiment.get(0);

     				

     				// perform program logic here

     				String output = tweet.get(0)+"/"+sentiment.get(0);

     				try {

                   	   

            			//String content = "This is the content to write into file";

             			

            			File file = new File("/home/cloudera/Desktop/ADTRead.txt");

             

            			// if file doesnt exists, then create it

            			if (!file.exists()) {

            				file.createNewFile();

            			}

             

            			FileWriter fw3 = new FileWriter(file.getAbsoluteFile(),true);

            			BufferedWriter bw3 = new BufferedWriter(fw3);

            			

            		   

                 			bw3.write(output);

                 			bw3.write("\n");

                 			bw3.close();

                  

                 			System.out.println("Done");

                  

                 		

                      }catch (IOException e) {

                 			e.printStackTrace();

                 		}

     				

     			}

     	

     			tweet.close();

     			sentiment.close();

     		

     		} catch (IOException e) {

     			e.printStackTrace();

     		}

          

    }

}    

