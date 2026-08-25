                             model model_kind      genre  support  precision   recall       f1  average_precision  predicted_positive
             Image Only (ResNet18)      image     Action     1078   0.521505 0.899814 0.660313           0.640112                1860
             Image Only (ResNet18)      image  Adventure      967   0.465544 0.915202 0.617155           0.592636                1901
             Image Only (ResNet18)      image     Casual     1084   0.510270 0.870849 0.643490           0.644905                1850
             Image Only (ResNet18)      image        RPG      484   0.358459 0.442149 0.395930           0.383875                 597
             Image Only (ResNet18)      image     Racing      201   0.513699 0.373134 0.432277           0.457937                 146
             Image Only (ResNet18)      image Simulation      624   0.425197 0.605769 0.499670           0.494036                 889
             Image Only (ResNet18)      image     Sports      210   0.530120 0.209524 0.300341           0.333209                  83
             Image Only (ResNet18)      image   Strategy      560   0.429245 0.487500 0.456522           0.461965                 636
            Text Only (DistilBERT)       text     Action     1078   0.621160 0.844156 0.715690           0.798445                1465
            Text Only (DistilBERT)       text  Adventure      967   0.585122 0.845915 0.691755           0.761048                1398
            Text Only (DistilBERT)       text     Casual     1084   0.617536 0.753690 0.678853           0.737160                1323
            Text Only (DistilBERT)       text        RPG      484   0.603774 0.528926 0.563877           0.607981                 424
            Text Only (DistilBERT)       text     Racing      201   0.744186 0.636816 0.686327           0.715508                 172
            Text Only (DistilBERT)       text Simulation      624   0.610200 0.536859 0.571185           0.652711                 549
            Text Only (DistilBERT)       text     Sports      210   0.689655 0.476190 0.563380           0.678252                 145
            Text Only (DistilBERT)       text   Strategy      560   0.660550 0.385714 0.487035           0.616873                 327
Multimodal (ResNet18 + DistilBERT) multimodal     Action     1078   0.651748 0.864564 0.743222           0.809163                1430
Multimodal (ResNet18 + DistilBERT) multimodal  Adventure      967   0.651952 0.794209 0.716084           0.763912                1178
Multimodal (ResNet18 + DistilBERT) multimodal     Casual     1084   0.565089 0.880996 0.688536           0.745791                1690
Multimodal (ResNet18 + DistilBERT) multimodal        RPG      484   0.592233 0.504132 0.544643           0.625206                 412
Multimodal (ResNet18 + DistilBERT) multimodal     Racing      201   0.642857 0.716418 0.677647           0.741690                 224
Multimodal (ResNet18 + DistilBERT) multimodal Simulation      624   0.629108 0.644231 0.636580           0.683639                 639
Multimodal (ResNet18 + DistilBERT) multimodal     Sports      210   0.612903 0.633333 0.622951           0.668410                 217
Multimodal (ResNet18 + DistilBERT) multimodal   Strategy      560   0.640091 0.501786 0.562563           0.651325                 439