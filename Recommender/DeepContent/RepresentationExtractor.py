import os
import librosa
import librosa.display

class RepresentationExtractor():

    def __init__(self):
        pass

    def extractBulk(self, dirPath):
        outputFiles = []

        print('Directory path: '+ dirPath)

        # walking through a directory to get do bulk time-frequency representation of the audio
        for root, subdirs, files in os.walk(dirPath):
            i = 0
            for filename in files:
                i = i + 1
                if filename.endswith('.wav'):
                    self._fileProgress(i, len(files), filename)

                    file_path = os.path.join(root, filename)
                    # ppFileName = rreplace(file_path, ".au", ".pp", 1)

                    try:
                        # prepossessingAudio(file_path, ppFileName)
                        outputFiles.append(_extract(file_path))
                    except Exception as e:
                        print("Error accured" + str(e))

                # if filename.endswith('au'):
                #     sys.stdout.write("\r%d%%" % int(i / 7620 * 100))
                #     sys.stdout.flush()
                #     i += 1


        return outputFiles

    def extractSingle(self, filePath):
        return self._extract(filePath)

    def _extract(self, filePath):
        y, sr = librosa.load(filePath)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfccs, x_axis='time')
        plt.colorbar()
        plt.title(filePath.split('/')[-1])
        plt.tight_layout()
        plt.show()
        outputFile = False
        return outputFile

    def _fileProgress(self, ith_file, total_files, filename):
        print('[{0}/{1}] {2}'.format(ith_file, total_files, filename))

    def _update_progress(self, progress):
        print('\r[{0}] {1}%'.format('#' * (progress / 10), progress))

if __name__ == '__main__':
    R = RepresentationExtractor()
    R.extractSingle('/home/capt4ce/projects/major_project/dataset/GTZAN_dataset/blues.00000.au')