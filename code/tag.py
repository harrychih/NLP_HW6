"""
Command-line interface for training and evaluating HMM and CRF taggers.
"""
import argparse
import logging
from pathlib import Path
from eval import model_cross_entropy, model_error_rate, tagger_write_output
from hmm import HiddenMarkovModel
from crf import CRFBiRNNModel
from lexicon import build_lexicon
from corpus import TaggedCorpus

logging.basicConfig(format="%(levelname)s : %(message)s", level=logging.INFO) 

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval", type=str, help="evalutation file")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="optional initial model file to load (will be trained further).  Loading a model overrides most of the other options."
    )
    parser.add_argument(
        "-l",
        "--lexicon",
        type=str,
        help="newly created model (if no model was loaded) should use this lexicon file",
    )
    parser.add_argument(
        "--crf",
        action="store_true",
        default=False,
        help="the newly created model (if no model was loaded) should be a CRF"
    )
    parser.add_argument(
        "--withBirnn",
        action="store_true",
        default=False,
        help="the newly created model (if no model was loaded) should be a CRF should be trained with Birnn feature selection"
    )
    parser.add_argument(
        "-u",
        "--unigram",
        action="store_true",
        default=False,
        help="the newly created model (if no model was loaded) should only be a unigram HMM or CRF"
    )
    parser.add_argument(
        "-a",
        "--awesome",
        action="store_true",
        default=False,
        help="the newly created model (if no model was loaded) should use extra improvements"
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str,
        nargs="*",
        help="training files to train the model further"
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=50000,
        help="maximum number of steps to train to prevent training for too long "
             "(this is an practical trick that you can choose implement in the `train` method of hmm.py and crf.py)"
    )
    parser.add_argument(
        "--save_step",
        type=int,
        default=5000,
        help="number of steps to save model once to prevent training for too long "
    )
    parser.add_argument(
        "--reg",
        type=float,
        default=1.0,
        help="l2 regularizamtion during further training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="learning rate during further training"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="tolerance for early stopping"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="tmp.model",
        help="where to save the trained model"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="where to save the prediction outputs"
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="logging_level", action="store_const", const=logging.WARNING
    )

    args = parser.parse_args()
    if not args.model and not args.lexicon:
        parser.error("Please provide lexicon file path when no model provided")
    if not args.model and not args.train:
        parser.error("Please provide at least one training file when no model provided")
    return args

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

    train = None
    model = None
    if args.model is not None:
        if args.crf:
            model = CRFBiRNNModel.load(Path(args.model))
        else:
            model = HiddenMarkovModel.load(Path(args.model))
        assert model is not None
        tagset = model.tagset
        vocab = model.vocab
        if args.train is not None:
            train = TaggedCorpus(*[Path(t) for t in args.train], tagset=tagset, vocab=vocab)
    else:
        train = TaggedCorpus(*[Path(t) for t in args.train])
        tagset = train.tagset
        vocab = train.vocab
        if args.crf:
            lexicon = build_lexicon(train, embeddings_file=Path(args.lexicon), log_counts=args.awesome)
            model = CRFBiRNNModel(tagset, vocab, lexicon, unigram=args.unigram, withBirnn=args.withBirnn)
        else:
            lexicon = build_lexicon(train, embeddings_file=Path(args.lexicon), log_counts=args.awesome)
            model = HiddenMarkovModel(tagset, vocab, lexicon, unigram=args.unigram)

    dev = TaggedCorpus(Path(args.eval), tagset=tagset, vocab=vocab)
    if args.train is not None:
        assert train is not None and model is not None
        # you can instantiate a different development loss depending on the question / which one optimizes performance
        dev_loss =  lambda x: model_cross_entropy(x, dev)
        try:
            model.train(corpus=train,
                        loss=dev_loss,
                        minibatch_size=args.train_batch_size,
                        evalbatch_size=args.eval_batch_size,
                        lr=args.lr,
                        reg=args.reg,
                        save_path=args.save_path,
                        tolerance=args.tolerance,
                        max_iter=args.max_iters,
                        save_step=args.save_step)
        except KeyboardInterrupt:
            logging.info(f"KeyboardInterrupt -- saved model to {args.save_path}")
            model.save(args.save_path)
    tagger_write_output(model, dev, Path(args.eval+".output") if args.output_file is None else args.output_file)


if __name__ == "__main__":
    main()
