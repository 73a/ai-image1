package jp.jpdirect.image.classification.learning2;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileGraphSaver;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LocalResponseNormalization;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.FaceNetNN4Small2;
import org.deeplearning4j.zoo.model.ResNet50;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;

import ereshkigal.commons.FILES;
import ereshkigal.commons.commandline.CommandLine;
import ereshkigal.commons.commandline.CommandLine.OptionParameter;
import ereshkigal.commons.commandline.CommandLine.SimpleParameter;
import ereshkigal.commons.logging.Logger;

public class ImageClassificationLearning2 {
	private static Logger logger = Logger.getInstance(ImageClassificationLearning2.class);

	private enum ModelName {
		alexnet("alexnet", "a"),
		renet("renet", "r"),
		resnet50("resnet50", "r50"),
		facenet("facenet", "f"),
		;
		private String label;
		private String alias;

		private ModelName(String label, String alias) {
			this.label = label;
			this.alias = alias;
		}

		public String label() {
			return label;
		}

		@SuppressWarnings("unused")
		public String alias() {
			return alias;
		}

		@SuppressWarnings("unused")
		public static ModelName fromLabel(String label) {
			for (ModelName type : ModelName.values()) {
				if (type.label.equals(label)) {
					return type;
				}
			}
			return null;
		}

		@SuppressWarnings("unused")
		public static ModelName fromAlias(String alias) {
			for (ModelName type : ModelName.values()) {
				if (type.alias.equals(alias)) {
					return type;
				}
			}
			return null;
		}

		public static ModelName fromLabelOrAlias(String labelOrAlias) {
			for (ModelName type : ModelName.values()) {
				if (type.label.equals(labelOrAlias) || type.alias.equals(labelOrAlias)) {
					return type;
				}
			}
			return null;
		}
	}

	public enum ModelType {
		MultiLayerNetwork("MultiLayerNetwork"),
		ComputationGraph("ComputationGraph"),
		;
		private String label;

		private ModelType(String label) {
			this.label = label;
		}

		public String label() {
			return label;
		}

		public static ModelType fromLabel(String label) {
			for (ModelType type : ModelType.values()) {
				if (type.label.equals(label)) {
					return type;
				}
			}
			return null;
		}
	}

	private static long DEFAULT_SEED = 42;

	protected ModelType modelType;

    protected long seed;
    protected Random rng = new Random(seed);
    protected int listenerFreq = 1;
    protected int maxPathsPerLabel = 0;

    protected List<String> labels;

    protected int width = 96;
    protected int height = 96;
    protected int channels = 3;
    protected int batchSize = 10;
    protected int iterations = 1;
    protected int epochs = 50;
    protected double splitTrainTest = 0.8d;

    protected String input;
    protected String output;
    protected String property;
    protected String label;
    protected String modelDirectory = "model.work";

    private ParentPathLabelGenerator labelMaker;
    private InputSplit trainData;
    private InputSplit testData;

	public ImageClassificationLearning2(long seed) {
		this.seed = seed;
	    this.rng = new Random(seed);
	}

	public int getHeight() {
		return height;
	}

	public ImageClassificationLearning2 setHeight(int height) {
		this.height = height;
		return this;
	}

	public int getWidth() {
		return width;
	}

	public ImageClassificationLearning2 setWidth(int width) {
		this.width = width;
		return this;
	}

	public int getChannels() {
		return channels;
	}

	public ImageClassificationLearning2 setChannels(int channels) {
		this.channels = channels;
		return this;
	}

	public int getBatchSize() {
		return batchSize;
	}

	public ImageClassificationLearning2 setBatchSize(int batchSize) {
		this.batchSize = batchSize;
		return this;
	}

	public int getListenerFreq() {
		return listenerFreq;
	}

	public ImageClassificationLearning2 setListenerFreq(int listenerFreq) {
		this.listenerFreq = listenerFreq;
		return this;
	}

	public int getIterations() {
		return iterations;
	}

	public ImageClassificationLearning2 setIterations(int iterations) {
		this.iterations = iterations;
		return this;
	}

	public int getEpochs() {
		return epochs;
	}

	public ImageClassificationLearning2 setEpochs(int epochs) {
		this.epochs = epochs;
		return this;
	}

	public double getSplitTrainTest() {
		return splitTrainTest;
	}

	public ImageClassificationLearning2 setSplitTrainTest(double splitTrainTest) {
		this.splitTrainTest = splitTrainTest;
		return this;
	}

	public String getOutput() {
		return output;
	}

	public ImageClassificationLearning2 setOutput(String output) {
		this.output = output;
		return this;
	}

	public String getProperty() {
		return property;
	}

	public ImageClassificationLearning2 setProperty(String property) {
		this.property = property;
		return this;
	}

	public String getInput() {
		return input;
	}

	public ImageClassificationLearning2 setInput(String input) {
		this.input = input;
		return this;
	}

	public String getLabel() {
		return label;
	}

	public ImageClassificationLearning2 setLabel(String label) {
		this.label = label;
		return this;
	}

	public String getModelDirectory() {
		return modelDirectory;
	}

	public ImageClassificationLearning2 setModelDirectory(String modelDirectory) {
		this.modelDirectory = modelDirectory;
		return this;
	}

	public void learnRenet() {
		modelType = ModelType.MultiLayerNetwork;
		setupData();
		learn(renet());
	}

	public void learnAlexnet() {
		modelType = ModelType.MultiLayerNetwork;
		setupData();
		learn(alexnet());
	}

	public void learnResnet50() {
		modelType = ModelType.ComputationGraph;
		setupData();
		learn(resnet50());
	}

	public void learnFacenet() {
		modelType = ModelType.ComputationGraph;
		setupData();
		learn(facenet());
	}

	private void setupData() {
		message("");

		message("Data Setup -> organize and limit data file paths");
		labelMaker = new ParentPathLabelGenerator();
        File mainPath = new File(input);
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
        getLabels(fileSplit.getRootDir());
        int numExamples = java.lang.Math.toIntExact(fileSplit.length());
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, labels.size(), maxPathsPerLabel);

        message("Data Setup -> train test split");
        InputSplit[] inputSplit = fileSplit.sample(pathFilter, splitTrainTest, 1 - splitTrainTest);
        trainData = inputSplit[0];
        testData = inputSplit[1];
	}

	private void learn(MultiLayerConfiguration networkConfig) {
        message("Data Setup -> normalization");
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

        message("Data Setup -> define how to load data into net");
        try(ImageRecordReader trainReader = new ImageRecordReader(height, width, channels, labelMaker);
        		ImageRecordReader testReader = new ImageRecordReader(height, width, channels, labelMaker)) {
        	trainReader.initialize(trainData, null);
        	testReader.initialize(testData, null);
        	DataSetIterator trainIter = new RecordReaderDataSetIterator(trainReader, batchSize, 1, labels.size());
        	trainIter.setPreProcessor(scaler);
        	DataSetIterator testIter = new RecordReaderDataSetIterator(testReader, batchSize, 1, labels.size());
        	testIter.setPreProcessor(scaler);


	        message("Build model....");
        	EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
        			.epochTerminationConditions(new MaxEpochsTerminationCondition(epochs))
        			.scoreCalculator(new DataSetLossCalculator(testIter, true))
        	        .evaluateEveryNEpochs(1)
        			.modelSaver(new LocalFileModelSaver(modelDirectory))
        			.build();

            message("Save model property....");
            saveProperty();

            message("Save label....");
            saveLabel();

        	EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, networkConfig, trainIter);

	        message("Train model....");
        	EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();
        	Model bestModel = result.getBestModel();

            message("Save model....");
            ModelSerializer.writeModel(bestModel, output, true);

            message("Finished");
        } catch (Exception e) {
        	e.printStackTrace();
        }
	}

	private void learn(ComputationGraph graphConfig) {
        message("Data Setup -> normalization");
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

        message("Data Setup -> define how to load data into net");
        try(ImageRecordReader trainReader = new ImageRecordReader(height, width, channels, labelMaker);
        		ImageRecordReader testReader = new ImageRecordReader(height, width, channels, labelMaker)) {
        	trainReader.initialize(trainData, null);
        	testReader.initialize(testData, null);
        	DataSetIterator trainIter = new RecordReaderDataSetIterator(trainReader, batchSize, 1, labels.size());
        	trainIter.setPreProcessor(scaler);
        	DataSetIterator testIter = new RecordReaderDataSetIterator(testReader, batchSize, 1, labels.size());
        	testIter.setPreProcessor(scaler);


	        message("Build model....");
        	EarlyStoppingConfiguration<ComputationGraph> esConf = new EarlyStoppingConfiguration.Builder<ComputationGraph>()
        			.epochTerminationConditions(new MaxEpochsTerminationCondition(epochs))
        			.scoreCalculator(new DataSetLossCalculator(testIter, true))
        	        .evaluateEveryNEpochs(1)
        			.modelSaver(new LocalFileGraphSaver(modelDirectory))
        			.build();

            message("Save model property....");
            saveProperty();

            message("Save label....");
            saveLabel();

        	EarlyStoppingGraphTrainer trainer = new EarlyStoppingGraphTrainer(esConf, graphConfig, trainIter);

	        message("Train model....");
        	EarlyStoppingResult<ComputationGraph> result = trainer.fit();
        	Model bestModel = result.getBestModel();

            message("Save model....");
            ModelSerializer.writeModel(bestModel, output, true);

            message("Finished");
        } catch (Exception e) {
        	e.printStackTrace();
        }
	}

	private MultiLayerConfiguration renet() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.005)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(0.9))
                .list()
                .layer(0, new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}).name("cnn1").nIn(channels).nOut(50).biasInit(0).build())
                .layer(1, new SubsamplingLayer.Builder(new int[]{5, 5}, new int[]{2, 2}).name("maxpool1").build())
                .layer(2, new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{5, 5}, new int[]{1, 1}).name("cnn2").nOut(200).biasInit(0).build())
                .layer(3, new SubsamplingLayer.Builder(new int[]{2, 2}, new int[]{2, 2}).name("maxpool2").build())
                .layer(4, new DenseLayer.Builder().nOut(1000).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(labels.size())
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true)
                .pretrain(false)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        return conf;
	}

	private MultiLayerConfiguration alexnet() {
		double nonZeroBias = 1;
		double dropOut = 0.5;

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
	            .seed(seed)
	            .weightInit(WeightInit.DISTRIBUTION)
	            .dist(new NormalDistribution(0.0, 0.01))
	            .activation(Activation.RELU)
	            .updater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 1e-2, 0.1, 100000), 0.9))
	            .biasUpdater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 2e-2, 0.1, 100000), 0.9))
	            .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
	            .l2(5 * 1e-4)
	            .list()
	            .layer(0, new ConvolutionLayer.Builder(new int[]{11, 11}, new int[]{4, 4}, new int[]{3, 3}).name("cnn1").nIn(channels).nOut(96).biasInit(0).build())
	            .layer(1, new LocalResponseNormalization.Builder().name("lrn1").build())
	            .layer(2, new SubsamplingLayer.Builder(new int[]{3,3}, new int[]{2,2}).name("maxpool1").build())
	            .layer(3, new ConvolutionLayer.Builder(new int[]{5,5}, new int[] {1,1}, new int[] {2,2}).name("cnn2").nOut(256).biasInit(nonZeroBias).build())
	            .layer(4, new LocalResponseNormalization.Builder().name("lrn2").build())
	            .layer(5, new SubsamplingLayer.Builder(new int[]{3,3}, new int[]{2,2}).name("maxpool2").build())
	            .layer(6, new ConvolutionLayer.Builder(new int[]{3,3}, new int[] {1,1}, new int[] {1,1}).name("cnn3").nOut(384).biasInit(0).build())
	            .layer(7, new ConvolutionLayer.Builder(new int[]{3,3}, new int[] {1,1}, new int[] {1,1}).name("cnn4").nOut(384).biasInit(nonZeroBias).build())
	            .layer(8, new ConvolutionLayer.Builder(new int[]{3,3}, new int[] {1,1}, new int[] {1,1}).name("cnn5").nOut(256).biasInit(nonZeroBias).build())
	            .layer(9, new SubsamplingLayer.Builder(new int[]{3,3}, new int[]{2,2}).name("maxpool3").build())
	            .layer(10, new DenseLayer.Builder().name("ffn1").nOut(4096).biasInit(nonZeroBias).dropOut(dropOut).dist(new GaussianDistribution(0, 0.005)).build())
	            .layer(11, new DenseLayer.Builder().name("ffn2").nOut(4096).biasInit(nonZeroBias).dropOut(dropOut).dist(new GaussianDistribution(0, 0.005)).build())
	            .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
	                .name("output")
	                .nOut(labels.size())
	                .activation(Activation.SOFTMAX)
	                .build())
	            .backprop(true)
	            .pretrain(false)
	            .setInputType(InputType.convolutional(height, width, channels))
	            .build();

	        return conf;
	}

	private ComputationGraph resnet50() {
		ResNet50 resnet50 = new ResNet50(
				seed,
				new int[] {3, 224, 224},
				labels.size(),
				WeightInit.DISTRIBUTION,
				new RmsProp(0.1, 0.96, 0.001),
				CacheMode.NONE,
				WorkspaceMode.ENABLED,
				ConvolutionLayer.AlgoMode.PREFER_FASTEST);
		ComputationGraph graph = resnet50.init();
		return graph;
	}

	private ComputationGraph facenet() {
		FaceNetNN4Small2 facenet = new FaceNetNN4Small2(
				seed,
				new int[] {3, 96, 96},
				labels.size(),
				new Adam(0.1, 0.9, 0.999, 0.01),
				Activation.RELU,
				CacheMode.NONE,
				WorkspaceMode.ENABLED,
				ConvolutionLayer.AlgoMode.PREFER_FASTEST,
				128
				);
		ComputationGraph model = facenet.init();
		message(model.summary());
		return model;
	}

	private void clearnModelDirectory() {
		Path dir = Paths.get(modelDirectory);
		if (Files.exists(dir)) {
			try {
				FILES.deleteDirectory(dir);
				Files.createDirectories(dir);
			} catch (IOException e) {
				logger.error(e);
			}
		} else {
			try {
				Files.createDirectories(dir);
			} catch (IOException e) {
				logger.error(e);
			}
		}
	}

	private void getLabels(File root) {
		this.labels = Arrays.stream(root.listFiles(File::isDirectory)).map(file->file.getName()).collect(Collectors.toList());
	}

	private void saveLabel() {
		try (BufferedWriter writer = Files.newBufferedWriter(Paths.get(label))) {
			for (String label : labels) {
				writer.write(label);
				writer.newLine();
			}
			writer.flush();
		} catch (IOException e) {
			message("ラベルの書き出し失敗。(" + label + "");
		}
	}

	private void saveProperty() {
		try (BufferedWriter writer = Files.newBufferedWriter(Paths.get(property))) {
			writer.write(modelType.label());
			writer.write(",");
			writer.write(String.valueOf(this.width));
			writer.write(",");
			writer.write(String.valueOf(this.height));
			writer.write(",");
			writer.write(String.valueOf(this.channels));
			writer.newLine();
		} catch (IOException e) {
			message("プロパティの書き出し失敗。(" + property + "");
		}
	}

	@SuppressWarnings("unused")
	private static void usage(String message) {
		message(message);
		usage();
	}

	private static void usage() {
		message("(this program) [<data directory>] [-nn [alexnet|renet|resnet50]] [-w <image width>] [-h <image height>] [-c <image channels>] [-b <batch size>] [-i <iterations>] [-e <epochs>] [-w <name of model working directory>] [-m <filename of learned model data>][-p <filename of model property] [-l <filename of labels>]");
	}

	private static void message(String message) {
		System.out.println(message);
	}

	public static void main(String...args) {
		CommandLine cl = CommandLine.parse(args);
		SimpleParameter[] sparam = cl.getSimpleParameter();
		OptionParameter param;

		if (cl.getOptionParameter("--help") != null) {
			usage();
			return;
		}

		ImageClassificationLearning2 icl = new ImageClassificationLearning2(DEFAULT_SEED);
		if (sparam.length == 1) {
			icl.setInput(sparam[0].getParameter());
			message("Data directory: " + sparam[0].getParameter());
		} else {
			icl.setInput("data");
			message("Data directory: data");
		}
		if (!Files.exists(Paths.get(icl.getInput()))) {
			message("No such directory: " + icl.getInput());
			return;
		}

		ModelName nnType = null;
		param = cl.getOptionParameter("-nn");
		if (param != null) {
			nnType = ModelName.fromLabelOrAlias(param.getParameter());
		}
		if (nnType == null) {
			nnType = ModelName.renet;
		}
		message("NN Model: " + nnType.label());

		param = cl.getOptionParameter("-w");
		if (param != null) {
			icl.setWidth(Integer.parseInt(param.getParameter()));
		}
		message("image width: " + icl.getWidth());

		param = cl.getOptionParameter("-h");
		if (param != null) {
			icl.setHeight(Integer.parseInt(param.getParameter()));
		}
		message("image height: " + icl.getHeight());

		param = cl.getOptionParameter("-c");
		if (param != null) {
			icl.setChannels(Integer.parseInt(param.getParameter()));
		}
		message("image channels: " + icl.getChannels());

	    param = cl.getOptionParameter("-b");
	    if (param != null) {
	    	icl.setBatchSize(Integer.parseInt(param.getParameter()));
	    }
	    message("batch size: " + icl.getBatchSize());

	    param = cl.getOptionParameter("-i");
	    if (param != null) {
	    	icl.setIterations(Integer.parseInt(param.getParameter()));
	    }
	    message("iterations: " + icl.getIterations());

	    param = cl.getOptionParameter("-e");
	    if (param != null) {
	    	icl.setEpochs(Integer.parseInt(param.getParameter()));
	    }
	    message("epochs: " + icl.getEpochs());

	    param = cl.getOptionParameter("-w");
	    if (param != null) {
	    	icl.setModelDirectory(param.getParameter());
	    }
	    message("model working directory: " + icl.getModelDirectory());
	    icl.clearnModelDirectory();

	    param = cl.getOptionParameter("-m");
	    if (param != null) {
	    	icl.setOutput(param.getParameter());
	    } else {
	    	icl.setOutput("learned." + nnType.label + ".bin");
	    }
	    message("learned model: " + icl.getOutput());

	    param = cl.getOptionParameter("-p");
	    if (param != null) {
	    	icl.setProperty(param.getParameter());
	    } else {
	    	icl.setProperty("learned." + nnType.label + ".prop");
	    }
	    message("model property: " + icl.getProperty());

	    param = cl.getOptionParameter("-l");
	    if (param != null) {
	    	icl.setLabel(param.getParameter());
	    } else {
	    	icl.setLabel("label.txt");
	    }
	    message("label: " + icl.getLabel());

	    switch (nnType) {
		case alexnet:
			icl.learnAlexnet();
			break;
		case renet:
			icl.learnRenet();
			break;
		case resnet50:
			icl.learnResnet50();
			break;
		case facenet:
			icl.learnFacenet();
			break;
		}
	}
}
