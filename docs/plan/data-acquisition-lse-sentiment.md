# Données à récupérer : London Strategic Edge et sentiment

Inventaire vérifié le 6 octobre 2026. Ce document fixe les données souhaitées, leur rôle et les conditions d'utilisation. Aucun téléchargement de corpus, installation ou changement de modèle n'est effectué à cette étape.

Le chantier suivant est décrit dans [le pilote sentiment](news-sentiment-pilot.md). Son premier essai authentifié a montré que `NEWS_SENTIMENT` est premium pour la clé utilisée ; la collecte a été arrêtée après un appel, sans article récupéré. Cette observation remplace l'hypothèse d'un accès gratuit à cet endpoint pour le pilote, sans changer le quota conservateur du collecteur.

LSE désigne ici **London Strategic Edge**, pas le London Stock Exchange. Les sources scientifiques restent dans [papers](../papers/), les interfaces du modèle dans [next steps](next_steps.md).

## Synthèse

LSE est retenu comme source candidate importante pour les données structurées. Son historique ne doit pas être considéré comme automatiquement point-in-time. Il faut prévoir une source complémentaire pour les versions historiques de la macro et une collecte séparée pour le sentiment.

Les familles ci-dessous sont documentées publiquement, mais leur couverture exacte, leurs droits et leurs horodatages ne sont pas encore validés sur des exports. Aucune clé API ni aucun échantillon authentifié n'a été utilisé. Les volumes globaux annoncés sur les pages LSE ne constituent pas une garantie par ticker ou série.

Le premier périmètre reste l'univers US du benchmark : 143 actions et 14 ETF de contexte. La période de référence visée va de 2005 à septembre 2026, sans garantie de cette profondeur pour chaque nouvelle source. Toute collecte plus récente forme une nouvelle version du dataset. Il faut conserver les fichiers figés des benchmarks et garder le holdout fermé lors du choix des données.

## 1. Inventaire des familles LSE

« Documenté » signifie qu'une famille ou une méthode existe, pas que toutes les variables souhaitées sont disponibles ni utilisables sans fuite future. Les usages et features proposés sont des pistes de travail, pas des résultats déjà démontrés.

| Famille documentée | Accès public documenté | Destination prévue | Priorité et condition |
| --- | --- | --- | --- |
| Prix actions et ETF | `candles()`, exports historiques | GRU, features de nœuds, contrôles du backtest | Socle. Vérifier instruments, séances, ajustements et provenance. |
| Indices et volatilité | Catalogue puis prix ou séries | Transformer marché | Prioritaire. Vérifier les identifiants exacts et la nature du prix. |
| Change et matières premières | Catalogue puis prix | Transformer marché | Prioritaire pour dollar, pétrole et or. |
| Taux souverains | `bond_yields()`, `series()` | Transformer marché | Prioritaire pour la courbe US. Horaires et conventions à auditer. |
| Séries macroéconomiques | `series()` | Transformer marché | Prioritaire seulement avec disponibilité historique et versions. |
| Calendrier économique | `economic_calendar()` | Événements macro | Vérifier consensus avant publication, première valeur et révisions. |
| États financiers structurés | `financial_reports()` | GRU, nœuds GNN | Prioritaire après audit des dépôts et retraitements. |
| Profils et fondamentaux | `company_profiles()`, `fundamentals()` | Référentiel, features d'entreprise | Snapshots : pas de remplissage historique automatique. |
| Dividendes et splits | `dividends()`, `splits()` | Normalisation des prix, événements | Socle de qualité. Date effective distincte de l'annonce. |
| Transactions d'initiés | `insider_trades()` | Features d'entreprise | Extension après validation de la disponibilité du dépôt. |
| Positionnement futures COT | `cot()` | Contexte marché | Extension après audit du calendrier réel de publication. |
| Options | `options()`, `options_flow()`, `option_candles()` | Risque implicite, contexte par actif | Exploration ciblée, pas de collecte exhaustive initiale. |
| Ticks et intraday | Streaming et exports | Gap, liquidité, étude après l'open | Optionnel, après validation des prix et du stockage. |

Sources : [SDK officiel, méthodes de lecture](https://github.com/londonstrategicedge/lse-data/blob/main/lse/client.py), [catalogue Databank](https://londonstrategicedge.com/data/), [offre API](https://londonstrategicedge.com/free-market-data-api/).

### Limites importantes de LSE

Les bougies actions/ETF sont annoncées ajustées des splits. L'horodatage d'une bougie désigne son début, pas sa disponibilité. Les séries macro exposées ne documentent pas de sélection `asof` ou de vintages. Un filtre historique de date n'est donc pas un filtre de disponibilité. Les fundamentals sont décrits comme des snapshots. [Documentation HTTP](https://londonstrategicedge.com/api-documentation/), [SDK](https://github.com/londonstrategicedge/lse-data/blob/main/lse/client.py).

La FAQ indique que des sources publiques sont traitées par leurs algorithmes et que des instruments proxy peuvent être utilisés. Il faut identifier les instruments réellement observés, les transformations et les sources sous-jacentes avant d'utiliser ces prix pour les labels ou les fills. Un proxy ne doit pas être traité comme une cotation exécutable de l'action. [FAQ LSE](https://londonstrategicedge.com/faq/).

Pour les options, il faut distinguer la chaîne actuelle, les prints et l'historique par contrat. La docstring du SDK décrit le flow sur la semaine récente, tandis que son README annonce aussi des prints historiques filtrables par date. Cette incohérence doit être résolue sur un échantillon. Il ne faut supposer ni chaîne historique complète ni Greeks observables sur toute la période 2005 à 2026. [SDK options](https://github.com/londonstrategicedge/lse-data/blob/main/lse/client.py), [README options](https://github.com/londonstrategicedge/lse-data#options).

## 2. Liste concrète des données souhaitées

Les symboles ci-dessous constituent la liste de collecte souhaitée. Leur présence chez LSE doit être confirmée dans son catalogue ; un identifiant Yahoo ou FRED n'est pas automatiquement un identifiant LSE.

### Prix et contexte de marché

| Bloc | Données à récupérer | Features envisagées |
| --- | --- | --- |
| Actions | OHLCV des 143 titres, devise, place, calendrier, splits, dividendes, conventions d'ajustement ; prix non ajustés si disponibles | Rendements, volatilité, volume relatif, liquidité, gap. |
| Marché et secteurs | `SPY`, `QQQ`, `IWM`, `XLB`, `VOX`, `XLE`, `XLF`, `XLI`, `XLK`, `XLP`, `VNQ`, `XLU`, `XLV`, `XLY` | Facteurs globaux/sectoriels, breadth, dispersion, rendements résiduels. |
| Volatilité | VIX ; VIX court/long terme et VVIX seulement si réellement disponibles | Niveau, variation, structure de volatilité. |
| Taux US | Échéances 3 mois, 2 ans, 5 ans, 10 ans et 30 ans | Pente 10 ans/2 ans, 10 ans/3 mois, mouvements de courbe. |
| Dollar et matières premières | Dollar index, EUR/USD, pétrole Brent/WTI, or | Contexte devise, inflation et rotation sectorielle. |
| Risque de crédit | Spread investment grade et high yield, ou indicateurs publics de stress financier | Stress de financement, régime risk-on/risk-off. |

Les 14 ETF correspondent au [contexte US existant](../../src/trading_system/data/us_market_context.py). `VOX` et `VNQ` sont les proxies déjà retenus ; ils ne doivent pas être remplacés silencieusement par `XLC` ou `XLRE`, dont les périodes disponibles diffèrent.

Pour les futures, il faut aussi récupérer le contrat, les échéances et la règle de roll. Une série continue reconstruite aujourd'hui peut modifier le passé. Pour les spreads de crédit, couverture et droits restent à confirmer : FRED annonce notamment une restriction à trois ans pour sa série ICE high yield à partir d'avril 2026. Un historique gratuit complet via cette route n'est donc pas garanti. [Notes de la série FRED](https://fred.stlouisfed.org/series/BAMLH0A0HYM2).

### Macro publiée et microéconomie des entreprises

| Bloc | Données à chercher | Condition indispensable |
| --- | --- | --- |
| Politique monétaire | Taux directeur Fed, décisions FOMC ; ECB/BoE pour une extension Europe | Date de décision, publication et version connue. |
| Inflation | CPI/core CPI, PCE/core PCE, PPI | Valeur initiale et révisions, pas seulement historique révisé actuel. |
| Activité et emploi | PIB, chômage, emploi non agricole, demandes d'allocations, ventes au détail, production industrielle | Période mesurée distincte de la publication ; unités et saisonnalité explicites. |
| Calendrier macro | Événement, horaire prévu, consensus, précédent, première valeur publiée | Historique du consensus prépublication et des corrections. |
| Comptes | Revenus, bénéfices/EPS, cash-flow opérationnel, capex, dette, cash, actifs, capitaux propres | Identifiant du dépôt, date de disponibilité, version/amendement. |
| Ratios dérivés | Croissance, marges, levier, valorisation, rendement du cash-flow | Calcul avec les comptes et le nombre d'actions connus à cette date. |
| Événements d'entreprise | Résultats, dividendes, splits, transactions d'initiés | Annonce/dépôt distingué de la période ou transaction. |
| Référentiel | Identifiant stable, ticker/CIK, secteur, industrie, changements de nom, cotation/radiation | Validité historique de chaque association. |

PMI, surprises de résultats, estimations d'analystes, historique des secteurs, composants d'indices, relations fournisseurs/clients et participations croisées sont des **besoins éventuels non confirmés chez LSE**. Ces données restent hors du socle tant que couverture, droits et disponibilité ne sont pas établis. Des données de sociétés radiées seraient utiles pour réduire le biais de survivance, mais leur présence reste à vérifier.

Il faut distinguer les variables d'entreprise de la microstructure : bid/ask, spread, transactions et carnet d'ordres sont une autre famille. Un carnet historique Level 2 et un volume d'options/open interest historique complet ne sont pas confirmés ici.

### Ordre de récupération retenu

1. Catalogue, schémas, couverture et petit échantillon de contrôle.
2. Prix/contexte quotidien et corporate actions, sans substitution immédiate au socle actuel.
3. Taux et macro PIT, en commençant par un petit panel US.
4. États financiers et référentiel d'entreprise PIT.
5. Sentiment historique admissible et journal de collecte prospectif, chantier indépendant.
6. Insiders, COT, options agrégées et intraday seulement après validation des étapes précédentes.

Ces priorités de données ne changent pas la numérotation des plans d'architecture ou des benchmarks.

## 3. Conditions point-in-time

Deux règles du contrat actuel restent à respecter : prix et graphes peuvent utiliser la clôture J pour une exécution ultérieure ; news et fondamentaux doivent respecter `available_at < J 00:00 UTC`. Les nouvelles publications macro suivent cette seconde règle tant qu'un autre protocole n'est pas explicitement créé.

| Date ou information | Ce qu'elle représente | Information à ne pas en déduire automatiquement |
| --- | --- | --- |
| `observation_date` / fin de période | Période économique ou comptable mesurée | Disponibilité de la valeur. |
| `published_at` / `accepted_at` | Publication annoncée ou acceptation d'un dépôt | Première disponibilité du contenu exact dans la source consommée. |
| `available_at` | Première disponibilité utilisable, appuyée par une preuve | Date reconstruite depuis la seule période mesurée. |
| `collected_at` / `observed_at` | Moment où le collecteur a effectivement vu la version | Disponibilité vingt ans auparavant. |
| Vintage / révision | Version de la valeur et intervalle où elle était connue | Autorisation de remplacer toutes les anciennes versions. |

Pour chaque source, il faut conserver identifiant, version, unité, devise, timezone, provenance, preuve de disponibilité, checksum et date de collecte. Une valeur ancienne ne doit pas être écrasée par un retraitement récent. Une date sans heure exige une règle conservatrice documentée, sans lui attribuer une heure précise non établie.

Pour la macro, **ALFRED est le complément retenu si LSE ne fournit pas les vintages**. Son API permet de retrouver les valeurs connues à une date passée, contrairement au mode FRED courant. La couverture de vintages doit être vérifiée série par série ; une date de publication ne garantit pas la disponibilité sur FRED à cette même heure. [ALFRED](https://alfred.stlouisfed.org/), [périodes real-time](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html), [dates de publication](https://fred.stlouisfed.org/docs/api/fred/release_dates.html).

Pour les insiders, il faut récupérer le dépôt disponible, pas seulement `transaction_date`. Pour COT, la date des positions est généralement le mardi alors que la publication intervient le vendredi, avec exceptions. Le calendrier effectif et les éventuelles révisions doivent être conservés ; un décalage fixe n'est pas une preuve complète. [Calendrier CFTC](https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm).

Les contrôles prix portent sur les premières cotations régulières, les clôtures réellement terminées, le DST et les séances des différentes places. Le timestamp de début d'une bougie journalière ne permet pas d'utiliser sa clôture à l'ouverture. Il faut maintenir la cohérence des conventions de prix, dividendes, ajustements, coûts et benchmark buy & hold. Une convention d'ajustement rétroactive doit être auditée pour les features dépendant des niveaux absolus.

Sans preuve PIT historique, la donnée doit être classée comme snapshot actuel ou historique exploratoire. Elle reste hors benchmark PIT strict. Commencer aujourd'hui un journal `observed_at` sert aux décisions futures, pas à rendre rétrospectivement PIT un backfill ancien.

Les [snapshots Yahoo horodatés du projet](../src/point-in-time-context.md) suivent déjà cette logique prospective. Ils ne remplacent pas les vintages macro ni les publications d'entreprise historiques à récupérer ici.

## 4. Sources de sentiment à récupérer

Le scoring FinBERT reste dans [news-sentiment-feature-engineering](https://github.com/loic-mmt/news-sentiment-feature-engineering). Le dépôt de trading consomme son export, il ne collecte pas le corpus et ne lance pas FinBERT pendant l'entraînement.

Aucun endpoint news, sentiment journalistique ou transcript n'a été trouvé dans le SDK et la documentation publique LSE consultés. Ce n'est pas la preuve qu'aucune offre privée existe, mais la collecte du corpus ne doit pas dépendre d'une offre non documentée. Les états financiers structurés ne remplacent pas des textes de news. [SDK LSE](https://github.com/londonstrategicedge/lse-data).

| Source candidate | Données à rechercher | Décision et limite |
| --- | --- | --- |
| SEC EDGAR | Textes 8-K, 10-K, 10-Q et communiqués annexés | Priorité historique US. Corpus de disclosures distinct de la presse. |
| GDELT GKG 2.0 | Archives de mentions, organisations, thèmes, URLs, tonalité | Inventaire de news depuis le 19 février 2015. Ce n'est pas un corpus de corps d'articles ni du FinBERT. |
| Alpha Vantage `NEWS_SENTIMENT` | News filtrées par ticker et dates | Complément à auditer : profondeur et disponibilité historique non garanties. |
| Alpha Vantage `EARNINGS_CALL_TRANSCRIPT` | Transcripts trimestriels, annoncés depuis 2010Q1 | Optionnel. Accès gratuit effectif, couverture et droits à confirmer. |
| RSS et relations investisseurs | Titres, descriptions, communiqués et publications accessibles | Collecte prospective pour constituer un journal de disponibilité. |

EDGAR offre un accès public gratuit et des index dès 1994Q3, avec au maximum dix requêtes par seconde. Il faut cibler les émetteurs nécessaires et leurs exhibits, pas toute l'archive. [Accès EDGAR](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data), [API SEC](https://www.sec.gov/search-filings/edgar-application-programming-interfaces).

Attention : la SEC précise que l'acceptation d'un filing n'est pas sa première disponibilité web, et qu'aucun timestamp n'indique cette dernière. Il ne faut donc pas poser automatiquement `available_at = acceptanceDateTime`. Il faut une preuve de diffusion/observation historique, sinon le texte reste exploratoire ou disponible seulement à partir de la collecte effectuée. Les corrections ultérieures doivent aussi être distinguées. [FAQ SEC sur les timestamps](https://www.sec.gov/files/about/webmaster-faq.htm).

Pour GDELT, une URL historique ne prouve pas que le texte téléchargé aujourd'hui est identique au texte d'origine. Il faut rechercher une archive du contenu exact avec preuve de disponibilité. La recherche plein texte de DOC ne livre pas automatiquement les corps des articles ; sa documentation décrit une fenêtre glissante et un plafond de 250 résultats, pas un backfill universel depuis 2005. La rétention actuelle n'a pas été testée. [GDELT 2.0](https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/), [DOC API](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/).

Alpha Vantage limite les réponses news à 1 000 résultats ; une fenêtre saturée doit être subdivisée et reste incomplète tant qu'elle n'est pas réconciliée. Le quota gratuit général annoncé est de 25 appels/jour, avec un accès augmenté pour certains projets vérifiés. Ce n'est pas une garantie d'accès gratuit à chaque endpoint. Il faut aussi vérifier les droits d'usage et les dates de disponibilité des transcripts. [Documentation](https://www.alphavantage.co/documentation/), [quotas](https://www.alphavantage.co/support/), [conditions](https://www.alphavantage.co/terms_of_service/).

L'existence d'un corpus gratuit, complet et PIT de presse US sur toute la période 2005 à 2026 n'est pas établie aujourd'hui. Les droits du contenu source restent à vérifier, même si ses métadonnées sont accessibles gratuitement. Il faut séparer les expériences sur titres, résumés, textes complets, filings et transcripts, sans les regrouper indistinctement sous le terme « news ».

### Livrables attendus de la librairie de sentiment

Le [bridge déjà présent](../src/news-sentiment-integration.md) attend un corpus scoré et un journal de couverture séparé. Il faut conserver ce contrat et ajouter les preuves de collecte dans le manifeste plutôt que de multiplier les adaptations dans le modèle.

| Livrable | Informations à conserver |
| --- | --- |
| Corpus scoré | `news_id`, ticker/émetteur, source, URL, titre/texte, `published_at`, `available_at`, `availability_kind`, `availability_reference`, probabilités FinBERT et score. |
| Provenance du texte | Type de contenu, langue, hash de la version exacte, collecte, droits, méthode de résolution ticker et déduplication. |
| Journal de couverture | Source, ticker, début/fin d'intervalle, `recorded_at`, statut et preuve. Pages/partitions manquantes et saturation enregistrées. |
| Export quotidien | Features par séance/ticker, fenêtre d'agrégation, `coverage_status`, `last_contributing_available_at` et statistiques nullable ; masques et indicateurs de présence construits ensuite par le bridge et le dataset. |
| Manifeste | Hashes corpus/couverture/export, sources requises, versions des collecteurs et du mapping, checkpoint FinBERT immuable, paramètres, bornes UTC exclusives. |

Une couverture `covered` signifie collecte complète **du flux déclaré et du périmètre de requête**, pas connaissance de toutes les news du monde. Sa preuve doit aussi être connue avant `J 00:00 UTC`, via `recorded_at` dans le contrat amont. Auditer aujourd'hui un backfill complet ne permet pas d'antidater ce journal. Une preuve d'archive historique et la date de son audit actuel restent deux informations distinctes. HTTP 200 avec zéro résultat ne suffit pas. Une coupure, une pagination inachevée ou un texte manquant produit `incomplete`/`unknown`.

Il faut garder trois cas distincts : intervalle couvert sans article, sentiment neutre observé, couverture inconnue. Seul le premier justifie un vrai `news_count=0`. Une couverture inconnue garde le masque faux, même avec quelques articles présents. Les exemples et validations se trouvent dans [les tests du bridge](../../tests/test_news_sentiment_bridge.py).

## 5. Stockage et intégration dans le modèle

Le PC GPU conserve les archives brutes et les textes. Le Mac ne reçoit que les panels quotidiens utiles, les features agrégées et les manifestes. Il faut commencer par une petite période et quelques émetteurs pour mesurer taille, qualité et quotas avant toute collecte longue. Les ticks, options et archives GDELT globales restent hors téléchargement initial.

| Jeu préparé | Contenu | Branche ou usage prévu |
| --- | --- | --- |
| Prix et événements audités | OHLCV, corporate actions, calendrier, provenance | GRU, nœuds, labels et backtest. |
| Contexte marché quotidien | ETF, volatilité, courbe, change, matières premières | Transformer marché indépendant. |
| Publications macro PIT | Valeurs, vintages, disponibilité, fraîcheur | Groupe macro optionnel du contexte marché. |
| Entreprises PIT | Comptes, ratios causaux, événements, secteurs datés | Groupe entreprise optionnel du GRU/GNN. |
| Sentiment quotidien | Export FinBERT, couverture et manifeste | Branche sentiment indépendante. |
| Microstructure agrégée | Features intraday auditées | Protocole après l'open distinct. |

Ces noms décrivent des livrables futurs, pas de nouveaux arguments CLI déjà disponibles. Il reste à créer l'adaptateur LSE et à raccorder les nouvelles familles au runner approprié. L'intégration sentiment existante ne signifie pas que tous les runners consomment déjà le corpus.

Chaque groupe reste activable séparément. GRU et GNN gardent leurs entrées indépendantes ; le Transformer lit le contexte global, sans remplacer le GRU. Le sentiment ne devient pas une condition obligatoire pour les autres branches. Une valeur manquante n'est pas transformée en donnée réelle par un remplissage à zéro ou un backfill.

Il faut conserver un indicateur de fraîcheur pour les publications lentes, une limite de péremption documentée et des masques par source. Tout forward-fill part d'une publication déjà admissible ; aucun backward-fill. Scalers, sélection de features et constructions apprises restent train-only.

L'ajout d'information ne démontre pas l'utilité du GNN ou du gate. Pour la suite, il faut conserver `identity` et GRU comme contrôles, et tester les familles une par une sur des dates et expositions comparables. Si le sentiment commence en 2015, toutes les variantes de sa comparaison utilisent la même sous-période admissible, pas un contrôle évalué depuis 2005.

## 6. Audit à effectuer avant la collecte

- [ ] Confirmer dans `/catalog` et `/meta` les instruments, séries, unités, fréquences et premières/dernières dates, puis figer ce catalogue.
- [ ] Vérifier avec LSE quelles valeurs sont directement observées, dérivées ou proxy, et leur provenance.
- [ ] Obtenir le schéma des exports macro/comptes/calendrier : disponibilité, premières valeurs, révisions et horodatages par ligne.
- [ ] Valider sur un échantillon prix/volumes/corporate actions, heures de séance et cohérence avec le socle actuel.
- [ ] Vérifier couverture des sociétés radiées, changements de ticker et classifications sectorielles historiques ; sinon conserver explicitement le biais de survivance.
- [ ] Vérifier les quotas réels du compte utilisé dans `/usage`, pas un chiffre promotionnel ou un exemple de documentation.
- [ ] Confirmer les droits de stockage, transformation, entraînement et publication des résultats/poids/features dérivés.
- [ ] Fixer le panel macro minimal et sa route de vintages LSE ou ALFRED.
- [ ] Fixer séparément le corpus disclosures, le corpus presse éventuel et les sources prospectives avec leur journal de couverture.
- [ ] Mesurer le volume du pilote, puis dimensionner la collecte sur le PC seulement.
- [ ] Produire les manifestes et les tests anti-lookahead avant tout benchmark utilisant les nouvelles données.

Les requêtes de lignes paginées sont plafonnées à 5 000 lignes dans le SDK ; ce plafond ne décrit pas les réponses de découverte du catalogue. Les exports historiques utilisent des jobs avec téléchargement et reprise ; les quotas d'exports et de volume doivent être lus pour le compte utilisé. Il ne faut pas appliquer automatiquement les « dix téléchargements par heure » de la page Databank à tous les appels API. [SDK d'exports](https://github.com/londonstrategicedge/lse-data/blob/main/lse/vault.py), [documentation des quotas](https://londonstrategicedge.com/api-documentation/#usage), [Databank](https://londonstrategicedge.com/data/).

LSE autorise des usages internes de recherche, trading et entraînement, mais encadre la redistribution ; l'autorisation concernant les dérivés publiés doit être clarifiée. Aucun corpus, Parquet fournisseur ou clé ne doit être poussé sur GitHub. Une interface publique affichant leurs données exige une vérification spécifique des droits. Accès gratuit et licence MIT du SDK ne rendent pas les données librement redistribuables. [Conditions LSE](https://londonstrategicedge.com/terms/), [README et licence du client](https://github.com/londonstrategicedge/lse-data).

Les prochains benchmarks d'architecture seront décidés séparément. Cet inventaire fixe les besoins et les contrôles de données, sans annoncer leur disponibilité ni leur gain de performance.
