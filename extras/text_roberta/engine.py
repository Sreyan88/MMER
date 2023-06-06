import time
import torch
from tqdm import tqdm
import torchnet as tnt
import torchmetrics
import sys


class Engine(object):
    """
    A basic training engine to handle training , validation and testing 
    """
    def __init__(self,state = {}):
        
        self.state = state
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if self._state('gpu') is None:
            self.state['gpu'] = torch.cuda.is_available()
        if self._state('start_epoch') is None:
            self.state['start_epoch'] = 0
        if 'max_epochs' not in state:
            self.state['max_epochs'] = 4
        if self._state('best_score') is None:
            self.state['best_score'] = 0
        #Mode : Can be train,val,test
        if self._state('mode') is None:
            self.state['mode'] = 'train'

        # meters
        self.state['meter_loss'] = tnt.meter.AverageValueMeter()
        self.state['accuracy'] = torchmetrics.Accuracy(num_classes=4,average='weighted')
        self.state['f1_score'] = torchmetrics.F1(num_classes=4,average='macro')
        # time measure
        self.state['batch_time'] = tnt.meter.AverageValueMeter()
        self.state['data_time'] = tnt.meter.AverageValueMeter()

        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean').cuda()

        
    
    def _state(self,key):
        if key in self.state:
            return self.state[key]

    def on_start_epoch(self):
        self.state['meter_loss'].reset()
        self.state['batch_time'].reset()
        self.state['data_time'].reset()
        self.state['accuracy'].reset()
        self.state['f1_score'].reset()
        self.state['model_output'] = []
        self.state['ground_truth'] = []
    
    def on_end_epoch(self):
        loss = self.state['meter_loss'].value()[0]
        acc = self.state['accuracy'].compute()
        f1 = 0

        if self.state['mode']=='train':
            print('Epoch: [{0}]\t'
                    'Loss {loss:.4f}'.format(self.state['epoch'], loss=loss))
        elif self.state['mode']=='val':
            self.state['f1_score'](torch.cat(self.state['model_output']),torch.cat(self.state['ground_truth']).unsqueeze(1).to(dtype=torch.int))
            f1 = self.state['f1_score'].compute()
            print('====== F1 for epoch ',self.state['epoch'],' :: ',f1)
            print('Validation : \t Loss {loss:.4f}'.format(loss=loss))
        else:
            self.state['f1_score'](torch.cat(self.state['model_output']),torch.cat(self.state['ground_truth']).unsqueeze(1).to(dtype=torch.int))
            f1 = self.state['f1_score'].compute()
            print('Test : \t Loss {loss:.4f}'.format(loss=loss))
        return {'loss':loss,'accuracy':acc,'f1_score':f1, 'best_score':self.state['best_score']}
    
    def on_start_batch(self):
        pass

    def on_end_batch(self):
        # record loss
        self.state['loss_batch'] = self.state['loss'].data
        self.state['meter_loss'].add(self.state['loss_batch'].cpu())
        logits = self.state['output'].cpu()
        preds = torch.argmax(logits,axis=1)
        #preds = torch.flatten(torch.round(torch.sigmoid(logits)))
        self.state['accuracy'](preds,self.state['target'])


    def on_forward(self,model,input_ids,token_type_ids,attention_masks,labels,optimizer=None,scheduler = None):

        # compute output
        self.state['output'] = model(input_ids = input_ids,token_type_ids = token_type_ids,attention_mask = attention_masks,labels = labels)
        labels = labels.type(torch.LongTensor).cuda()

        self.state['loss'] = self.criterion(self.state['output'],labels)

        #self.state['loss'] = self.state['output']['loss']
        logits = self.state['output'].cpu()
        #preds = torch.flatten(torch.round(torch.sigmoid(logits)))
        preds = torch.argmax(logits,axis=1)


        if self.state['mode']=='train':
            self.state['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        if self.state['mode']=='val':
            self.state['model_output'].append(preds)
            self.state['ground_truth'].append(self.state['target'])
        if self.state['mode']=='test':
            self.state['model_output'].append(preds)
            self.state['ground_truth'].append(self.state['target'])
        
    

    def learn(self,train_loader,val_loader,model,opt,sch,test_loader = None,trial=None):
        '''
        Perform Training Operation
        '''
        #Send everything to device
        model.to(self.device)
        if self.state['gpu']:
          model.cuda()

        for epoch in range(self.state['start_epoch'],self.state['max_epochs']):
            
            self.state['epoch'] = epoch

            self.train_one_epoch(train_loader, model,opt,sch)

            metrics_dict = self.evaluate(val_loader, model)

            f1_score = metrics_dict['f1_score']
            acc = metrics_dict['accuracy']
            print('current f1 :: ',f1_score)
            print('accuracy ::',acc)
            self.state['best_score'] = max(self.state['best_score'],acc)
            if trial is not None:
                trial.report(self.state['best_score'], epoch)
            print(' *** best acc score ={best:.3f}'.format(best=self.state['best_score']))
        
        if test_loader is not None:
            print("============ Test Staring =================")
            metrics_dict = self.test(test_loader,model)
            return metrics_dict

    
    
    def train_one_epoch(self,train_loader, model, opt,sch):
        '''
        Train one Epoch
        '''
        
        train_loader = tqdm(train_loader,"Training Loader")
        self.on_start_epoch()

        self.state['mode'] = 'train'        
        model.train()
        
        end = time.time()


        for batch_idx,batch in enumerate(train_loader):
            
            self.state['iteration'] = batch_idx
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            

            self.state['input'] = batch[:2]
            self.state['target'] = batch[2]

            input_ids = self.state['input'][0].to(self.device)
            # token_type_ids = self.state['input'][1].to(self.device)
            token_type_ids = None
            attention_masks = self.state['input'][1].to(self.device)
            labels =  batch[2].to(self.device)

            self.on_start_batch()

            opt.zero_grad()

            self.on_forward(model,input_ids,token_type_ids,attention_masks,labels,opt,sch)
            
            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()


            # measure accuracy
            self.on_end_batch()


        self.on_end_epoch()


    def evaluate(self,val_loader, model):
        '''
        Evaluate the training metrics 
        '''
        val_loader = tqdm(val_loader,"Validation Loader")
        self.state['mode'] = 'val'
        self.on_start_epoch()

        model.eval()

        end = time.time()
        for batch_idx,batch in enumerate(val_loader):
            
            self.on_start_batch()
            with torch.no_grad():

                self.state['iteration'] = batch_idx
                self.state['data_time_batch'] = time.time() - end
                self.state['data_time'].add(self.state['data_time_batch'])

                self.state['input'] = batch[:2]
                self.state['target'] = batch[2]
                
                input_ids = self.state['input'][0].to(self.device)
                # token_type_ids = self.state['input'][1].to(self.device)
                token_type_ids = None
                attention_masks = self.state['input'][1].to(self.device)
                labels = batch[2].to(self.device)
                
                self.on_forward(model,input_ids,token_type_ids,attention_masks,labels)
                # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            # measure accuracy
            self.on_end_batch()


        metrics_dict = self.on_end_epoch()   

        return metrics_dict

    def test(self,test_loader, model):
        '''
        Evaluate the training metrics 
        '''
        test_loader = tqdm(test_loader,"Test Loader")
        self.state['mode'] = 'test'
        self.on_start_epoch()

        model.eval()

        end = time.time()
        for batch_idx,batch in enumerate(test_loader):
            
            self.on_start_batch()
            with torch.no_grad():

                self.state['iteration'] = batch_idx
                self.state['data_time_batch'] = time.time() - end
                self.state['data_time'].add(self.state['data_time_batch'])

                self.state['input'] = batch[:2]
                self.state['target'] = batch[2]
                
                input_ids = self.state['input'][0].to(self.device)
                # token_type_ids = self.state['input'][1].to(self.device)
                token_type_ids = None
                attention_masks = self.state['input'][1].to(self.device)
                labels = batch[2].to(self.device)

                self.on_forward(model,input_ids,token_type_ids,attention_masks,labels)
                # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            # measure accuracy
            self.on_end_batch()


        metrics_dict = self.on_end_epoch()   

        return metrics_dict

